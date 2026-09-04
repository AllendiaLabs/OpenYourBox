#pragma once

#include "../library/TrainingConfigLibrary.h"
#include "../graph/GraphTypes.h"
#include "../train/CloudTrainClient.h"
#include "../train/TrainCoordinator.h"

#include <imgui.h>
#include <JuceHeader.h>

#include <functional>
#include <string>
#include <vector>

namespace openyourbox::ui {
/**
 * @class TrainPanel
 * @brief Master-only ml_forge-style Run/Stop panel with live loss.
 */
class TrainPanel {
public:
  /** @brief User-editable recipe values sent with each Run request. */
  struct Hyperparameters {
    /** @brief Adam optimization steps. */
    int totalSteps = graph::defaultTrainSteps;
    /** @brief Initial learning rate before the 80%/95% decays. */
    float learningRate = graph::defaultTrainLearningRate;
    /** @brief RF-aware crop length in samples. */
    int segmentLength = graph::defaultTrainSegmentLength;
    /** @brief Steps between hear-while-training checkpoint exports. */
    int checkpointInterval = graph::defaultTrainCheckpointInterval;
    /** @brief Acids-rave v1 representation-stage steps. */
    int stage1Steps = graph::defaultReconstructionStage1Steps;
    /** @brief Acids-rave v1 quality-stage steps. */
    int stage2Steps = graph::defaultReconstructionStage2Steps;
    /** @brief Reconstruction crop length in samples. */
    int reconstructionSegmentLength = graph::defaultReconstructionSegmentLength;
    /** @brief Reconstruction minibatch size. */
    int batchSize = graph::defaultReconstructionBatchSize;
    /** @brief Generator Adam learning rate. */
    float generatorLr = graph::defaultReconstructionGeneratorLr;
    /** @brief Discriminator Adam learning rate. */
    float discriminatorLr = graph::defaultReconstructionDiscriminatorLr;
    /** @brief Adam β1 for generator and discriminator. */
    float adamBeta1 = graph::defaultReconstructionAdamBeta1;
    /** @brief Adam β2 for generator and discriminator. */
    float adamBeta2 = graph::defaultReconstructionAdamBeta2;
    /** @brief LinearLR end factor applied across stage 1. */
    float lrDecayEndFactor = graph::defaultReconstructionLrDecayEnd;
    /** @brief Target / constant KL β (acids-rave v1 = 0.1). */
    float klBeta = graph::defaultReconstructionKlBeta;
    /** @brief KL β at warmup start (v1 equals @ref klBeta). */
    float klBetaStart = graph::defaultReconstructionKlBetaStart;
    /** @brief KL warmup length in steps; 1 is constant β. */
    int klWarmupSteps = graph::defaultReconstructionKlWarmupSteps;
    /** @brief Feature-matching loss weight. */
    float featureMatchingWeight = graph::defaultReconstructionFeatureMatchingWeight;
    /** @brief Quality-stage discriminator update period. */
    int updateDiscriminatorEvery = graph::defaultReconstructionDiscUpdateEvery;
    /** @brief Probability of the v1 random allpass augmentation. */
    float phaseMangleProb = graph::defaultReconstructionPhaseMangleProb;
    /** @brief Dequantization bits; 0 disables the augmentation. */
    int dequantizeBits = graph::defaultReconstructionDequantizeBits;
    /** @brief Accelerator sent as `train_options.device`. */
    graph::TrainDevice device = graph::TrainDevice::automatic;
  };

  /** @brief Message-thread actions emitted by the Train panel. */
  struct Callbacks {
    /** @brief Start a job when copyright, selection, and arm gates pass. */
    std::function<void()> run;
    /** @brief Stop without auto-load. */
    std::function<void()> stop;
    /** @brief Retry loading a successful artifact after a swap failure. */
    std::function<void()> retryLoad;
    /** @brief Open the Training Library panel. */
    std::function<void()> openLibrary;
    /** @brief Open the copyright modal when acknowledgment is missing. */
    std::function<void()> requestCopyright;
    /** @brief Hear-while-training checkbox changed. */
    std::function<void(bool)> hearWhileTrainingChanged;
    /** @brief Opens the copyable dialog for the last train failure. */
    std::function<void(const juce::String &)> viewError;
    /** @brief Destination Local|Allendia changed. */
    std::function<void(graph::TrainDestination)> destinationChanged;
    /** @brief Launch Allendia account page from an entitlement refusal. */
    std::function<void()> openStorefront;
    /** @brief Download a remote checkpoint for hear-while-training. */
    std::function<void(const juce::String &)> downloadCheckpoint;
    /** @brief Persist current train settings on the project snapshot. */
    std::function<void(const juce::var &)> saveProjectConfig;
    /** @brief Active Data Loader picker changed. */
    std::function<void(std::int32_t)> activeDataLoaderChanged;
    /** @brief Download and load a cloud artifact from another machine. */
    std::function<void()> manualCloudLoad;
    /**
     * @brief Load a shipped example template (`mapping` or `reconstruction`).
     * Loads the companion graph and training-config example; not a Train mode.
     */
    std::function<void(const juce::String &)> loadExampleTemplate;
  };

  /** @brief Snapshot of Train enablement gates. */
  struct Gates {
    /** @brief Local copyright acknowledgment is on file. */
    bool copyrightAcknowledged = false;
    /** @brief Number of library pairs selected for this run. */
    int selectedPairCount = 0;
    /** @brief Total selected duration in seconds. */
    double selectedDurationSeconds = 0.0;
    /** @brief Sum of selected library file bytes (soft Cloud warn). */
    juce::int64 selectedUploadBytes = 0;
    /** @brief Soft Cloud upload warning threshold. */
    juce::int64 softUploadWarnBytes = graph::defaultCloudSoftUploadWarnBytes;
    /** @brief Number of armed trainable graph elements. */
    int armedElementCount = 0;
    /** @brief True when selected pairs mix sample rates (v1 block). */
    bool mixedSampleRates = false;
    /** @brief User-facing mixed-rate or empty-selection reason. */
    juce::String blockReason;
    /** @brief Informational receptive-field duration in milliseconds. */
    double receptiveFieldMilliseconds = 0.0;
    /** @brief Informational train-window duration in seconds. */
    double trainWindowSeconds = 0.0;
    /** @brief True when this instance is the capture/train master. */
    bool isMaster = true;
    /** @brief True when a train artifact exists but graph swap failed. */
    bool retryAvailable = false;
    /** @brief Mapping is blocked because unpaired clips are selected. */
    bool unpairedSelected = false;
    /** @brief Data Loader nodes on the canvas (id, label) for the active picker. */
    std::vector<std::pair<std::int32_t, juce::String>> dataLoaders;
    /** @brief Loss nodes on the canvas (id, label) for the stage editor. */
    std::vector<std::pair<std::int32_t, juce::String>> lossNodes;
    /**
     * @brief One row in the per-stage freeze structure tree (project structure).
     */
    struct FreezeStructureNode {
      /** @brief Graph node or group id. */
      std::int32_t id = 0;
      /** @brief Display label. */
      juce::String label;
      /** @brief True when this row is a group container. */
      bool isGroup = false;
      /**
       * @brief Leaf: currently armed. Group: any armed armable leaf under it.
       * Disarmed armable leaves are force-frozen (checked, gray, disabled).
       */
      bool armedForTraining = false;
      /** @brief Ordered children for groups. */
      std::vector<FreezeStructureNode> children;
    };
    /** @brief Root freeze-tree rows mirroring Project Structure. */
    std::vector<FreezeStructureNode> freezeStructure;
    /** @brief Platform account is linked. */
    bool cloudLinked = false;
    /** @brief Last entitlement probe is insufficient. */
    bool entitlementUnavailable = false;
    /** @brief Cloud poll lost the network without cancelling the job. */
    bool cloudOffline = false;
    /** @brief This instance submitted the attached cloud job. */
    bool isCloudSubmitter = true;
    /** @brief Non-submitter may download/load the finished artifact. */
    bool manualCloudLoadAvailable = false;
    /** @brief Published remote checkpoints for the attached job. */
    std::vector<train::CloudCheckpointInfo> cloudCheckpoints;
  };

  /**
   * @brief Draws Run/Stop, loss plot, hyperparameters, and recipe info.
   * @param coordinator Live train coordinator.
   * @param gates Enablement snapshot from the editor.
   * @param callbacks Editor-owned actions.
   */
  void render(const train::TrainCoordinator &coordinator, const Gates &gates,
              const Callbacks &callbacks);

  /**
   * @brief Fills MLflow experiment and tracking URI defaults.
   */
  TrainPanel();

  /** @brief Recipe values edited in the panel and read when Run is pressed. */
  Hyperparameters hyperparameters;
  /** @brief One Loss box selected for a stage, with its schedule weight. */
  struct LossStageEntry {
    /** @brief Canvas Loss node id. */
    std::int32_t lossNodeId = 0;
    /** @brief Multiplier applied to that Loss during the stage. */
    float weight = graph::defaultLossWeight;
  };

  /**
   * @brief Ordered loss stage schedule (empty = single stage of all wired losses).
   */
  struct LossStageDraft {
    /** @brief Optional stage name. */
    char name[64]{};
    /** @brief Stage duration in optimizer steps. */
    int steps = 1000;
    /** @brief Losses participating in this stage (weights live here, not on the box). */
    std::vector<LossStageEntry> losses;
    /**
     * @brief Element ids frozen for this stage (no grad); unarmed leaves stay listed.
     */
    std::vector<std::int32_t> freezeElementIds;
  };
  /** @brief User-edited multi-stage schedule; empty means use total_steps. */
  std::vector<LossStageDraft> lossStages;
  /** @brief When true, live audio swaps in exported training checkpoints. */
  bool hearWhileTraining = false;
  /** @brief When true, the train worker logs the run to an MLflow server. */
  bool logToMlflow = true;
  /** @brief MLflow tracking server origin, e.g. http://127.0.0.1:5000. */
  char mlflowTrackingUri[256]{};
  /** @brief MLflow experiment name passed to experiments/create. */
  char mlflowExperiment[128]{};
  /** @brief Optional MLflow run name; empty lets the server assign one. */
  char mlflowRunName[64]{};
  /** @brief Last-used Local vs Allendia destination (default local). */
  graph::TrainDestination destination = graph::TrainDestination::local;
  /** @brief Active Data Loader node id for the next Run, or 0 when unset. */
  std::int32_t activeDataLoaderId = 0;

  /**
   * @brief Serializes the current HP + stage editor to a JSON object.
   */
  [[nodiscard]] juce::var captureConfig() const;

  /**
   * @brief Applies recognized config fields; unknown keys are ignored.
   * @param config JSON object.
   * @return True when any recognized field was missing and defaults were used.
   */
  bool applyConfig(const juce::var &config);

  /** @brief User-level training-config catalog. */
  library::TrainingConfigLibrary configLibrary;
  /** @brief Save-as name buffer. */
  char configName[128]{"untitled"};
  /** @brief Warning shown after a forward-compatible load. */
  juce::String configLoadWarning;
};
} // namespace openyourbox::ui
