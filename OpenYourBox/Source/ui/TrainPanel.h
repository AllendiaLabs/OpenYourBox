#pragma once

#include "../graph/GraphTypes.h"
#include "../train/TrainCoordinator.h"

#include <imgui.h>
#include <JuceHeader.h>

#include <functional>
#include <string>
#include <vector>

namespace openyourbox::ui {
/**
 * @class TrainPanel
 * @brief Master-only ml_forge-style Run/Pause/Stop panel with live loss.
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
    /** @brief Pause the active job. */
    std::function<void()> pause;
    /** @brief Resume a paused job. */
    std::function<void()> resume;
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
  };

  /** @brief Snapshot of Train enablement gates. */
  struct Gates {
    /** @brief Local copyright acknowledgment is on file. */
    bool copyrightAcknowledged = false;
    /** @brief Number of library pairs selected for this run. */
    int selectedPairCount = 0;
    /** @brief Total selected duration in seconds. */
    double selectedDurationSeconds = 0.0;
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
    /** @brief Reconstruction graph is missing bottleneck/decode path. */
    bool reconstructionPathInvalid = false;
    /** @brief Reconstruction gate text. */
    juce::String reconstructionReason;
  };

  /**
   * @brief Draws Run/Pause/Stop, loss plot, hyperparameters, and recipe info.
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
  /** @brief Last-used Train objective for this plug-in instance. */
  graph::TrainObjective objective = graph::TrainObjective::mapping;
};
} // namespace openyourbox::ui
