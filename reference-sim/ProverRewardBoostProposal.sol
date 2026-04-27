pragma solidity >=0.8.27;

type Epoch is uint256;

function neqEpoch(Epoch _a, Epoch _b) pure returns (bool) {
    return Epoch.unwrap(_a) != Epoch.unwrap(_b);
}

using {neqEpoch as !=} for Epoch global;

// CompressedTimeMath.sol (aztec-packages/l1-contracts/src/shared/libraries/CompressedTimeMath.sol)
type CompressedEpoch is uint32;

library CompressedTimeMath {
    function compress(Epoch _epoch) internal pure returns (CompressedEpoch) {
        return CompressedEpoch.wrap(uint32(Epoch.unwrap(_epoch)));
    }

    function decompress(CompressedEpoch _epoch) internal pure returns (Epoch) {
        return Epoch.wrap(uint256(CompressedEpoch.unwrap(_epoch)));
    }
}

// SafeCast — OpenZeppelin (commit 448efeea)
library SafeCast {
    error SafeCastOverflowedUintDowncast(uint8 bits, uint256 value);

    function toUint(bool b) internal pure returns (uint256 u) {
        assembly ("memory-safe") {
            u := iszero(iszero(b))
        }
    }

    function toUint32(uint256 value) internal pure returns (uint32) {
        if (value > type(uint32).max) {
            revert SafeCastOverflowedUintDowncast(32, value);
        }
        return uint32(value);
    }
}

// Math — OpenZeppelin (commit 448efeea)
library Math {
    function ternary(bool condition, uint256 a, uint256 b) internal pure returns (uint256) {
        unchecked {
            return b ^ ((a ^ b) * SafeCast.toUint(condition));
        }
    }

    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return ternary(a > b, a, b);
    }

    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return ternary(a < b, a, b);
    }
}

// Errors (aztec-packages/l1-contracts/src/core/libraries/Errors.sol)
library Errors {
    error RewardBooster__OnlyRollup(address caller);
}

// IValidatorSelection (minimal — only getCurrentEpoch needed by RewardBooster)
interface IValidatorSelection {
    function getCurrentEpoch() external view returns (Epoch);
}

// ── RewardBooster types ───────────────────────────────────────────────────

struct RewardBoostConfig {
    uint32 increment;
    uint32 maxScore;
    uint32 a;
    uint32 minimum;
    uint32 k;
}

struct ActivityScore {
    Epoch time;
    uint32 value;
}

struct CompressedActivityScore {
    CompressedEpoch time;
    uint32 value;
}

interface IBoosterCore {
    function updateAndGetShares(address _prover) external returns (uint256);
    function getSharesFor(address _prover) external view returns (uint256);
}

interface IBooster is IBoosterCore {
    function getConfig() external view returns (RewardBoostConfig memory);
    function getActivityScore(address _prover) external view returns (ActivityScore memory);
}

// ── Full RewardBooster (from aztec-packages) ──────────────────────────────

/**
 * @title RewardBooster
 * @notice Abstracts the accounting related to rewards boosting from the POV of the rollup.
 * @dev    Source: aztec-packages/l1-contracts/src/core/reward-boost/RewardBooster.sol
 *         Dependencies inlined for self-contained governance deployment.
 */
contract RewardBooster is IBooster {
    using SafeCast for uint256;
    using CompressedTimeMath for Epoch;
    using CompressedTimeMath for CompressedEpoch;

    IValidatorSelection public immutable ROLLUP;
    uint256 private immutable CONFIG_INCREMENT;
    uint256 private immutable CONFIG_MAX_SCORE;
    uint256 private immutable CONFIG_A;
    uint256 private immutable CONFIG_MINIMUM;
    uint256 private immutable CONFIG_K;

    mapping(address prover => CompressedActivityScore) internal activityScores;

    modifier onlyRollup() {
        require(msg.sender == address(ROLLUP), Errors.RewardBooster__OnlyRollup(msg.sender));
        _;
    }

    constructor(IValidatorSelection _rollup, RewardBoostConfig memory _config) {
        ROLLUP = _rollup;

        CONFIG_INCREMENT = _config.increment;
        CONFIG_MAX_SCORE = _config.maxScore;
        CONFIG_A = _config.a;
        CONFIG_MINIMUM = _config.minimum;
        CONFIG_K = _config.k;
    }

    function updateAndGetShares(address _prover) external override(IBoosterCore) onlyRollup returns (uint256) {
        Epoch currentEpoch = ROLLUP.getCurrentEpoch();

        CompressedActivityScore storage store = activityScores[_prover];
        ActivityScore memory curr = _activityScoreAt(store, currentEpoch);

        if (curr.time != store.time.decompress()) {
            store.value = Math.min(curr.value + CONFIG_INCREMENT, CONFIG_MAX_SCORE).toUint32();
            store.time = curr.time.compress();
        }

        return _toShares(store.value);
    }

    function getConfig() external view override(IBooster) returns (RewardBoostConfig memory) {
        return RewardBoostConfig({
            increment: CONFIG_INCREMENT.toUint32(),
            maxScore: CONFIG_MAX_SCORE.toUint32(),
            a: CONFIG_A.toUint32(),
            minimum: CONFIG_MINIMUM.toUint32(),
            k: CONFIG_K.toUint32()
        });
    }

    function getSharesFor(address _prover) external view override(IBoosterCore) returns (uint256) {
        return _toShares(getActivityScore(_prover).value);
    }

    function getActivityScore(address _prover) public view override(IBooster) returns (ActivityScore memory) {
        return _activityScoreAt(activityScores[_prover], ROLLUP.getCurrentEpoch());
    }

    function _activityScoreAt(CompressedActivityScore storage _score, Epoch _epoch)
        internal
        view
        returns (ActivityScore memory)
    {
        uint256 decrease = (Epoch.unwrap(_epoch) - Epoch.unwrap(_score.time.decompress())) * 1e5;
        return ActivityScore({
            value: decrease > uint256(_score.value) ? 0 : _score.value - decrease.toUint32(),
            time: _epoch
        });
    }

    function _toShares(uint256 _value) internal view returns (uint256) {
        if (_value >= CONFIG_MAX_SCORE) {
            return CONFIG_K;
        }
        uint256 t = (CONFIG_MAX_SCORE - _value);
        uint256 rhs = CONFIG_A * t * t / 1e10;

        if (CONFIG_K < rhs) {
            return CONFIG_MINIMUM;
        }

        return Math.max(CONFIG_K - rhs, CONFIG_MINIMUM);
    }
}

// ── Governance Interfaces ─────────────────────────────────────────────────

interface IPayload {
    struct Action {
        address target;
        bytes data;
    }

    function getActions() external returns (Action[] memory);
}

interface IRollupCore {
    struct RewardConfig {
        address rewardDistributor;
        uint16 sequencerBps;
        address booster;
        uint96 checkpointReward;
    }

    function setRewardConfig(RewardConfig memory _config) external;
    function getRewardConfig() external view returns (RewardConfig memory);
}

// ── Governance Payload ────────────────────────────────────────────────────

contract ProverRewardBoostProposal is IPayload {
    /// @notice The Aztec Rollup contract on L1
    address public constant ROLLUP = 0x603bb2c05D474794ea97805e8De69bCcFb3bCA12;

    /// @notice The new RewardBooster deployed with proposed parameters
    address public immutable newBooster;

    constructor() {
        // Deploy new RewardBooster with proposed parameters
        RewardBoostConfig memory proposedConfig = RewardBoostConfig({
            increment: 101_400,   // proof_increase (was 125,000)
            maxScore:  367_500,   // score cap & threshold (was 15,000,000)
            a:         250_000,   // quadratic coefficient (was 1,000)
            minimum:    10_000,   // min shares floor (was 100,000)
            k:       1_000_000    // max shares (unchanged)
        });

        newBooster = address(new RewardBooster(
            IValidatorSelection(ROLLUP),
            proposedConfig
        ));
    }

    /// @notice Returns the governance actions to execute
    /// @dev Reads current config live to avoid TOCTOU — only overwrites the booster field
    function getActions() external view override returns (Action[] memory) {
        Action[] memory actions = new Action[](1);

        IRollupCore.RewardConfig memory cfg = IRollupCore(ROLLUP).getRewardConfig();
        cfg.booster = newBooster;

        actions[0] = Action({
            target: ROLLUP,
            data: abi.encodeWithSelector(IRollupCore.setRewardConfig.selector, cfg)
        });

        return actions;
    }
}
