#include "LiveGraphPublisher.h"

#include <utility>

namespace openyourbox::dsp {
LiveGraphPublisher::LiveGraphPublisher()
    : Thread("OpenYourBox Live Graph Compiler") {
  startThread();
}

LiveGraphPublisher::~LiveGraphPublisher() {
  signalThreadShouldExit();
  notify();
  stopThread(10000);
}

void LiveGraphPublisher::requestCompile(
    const juce::ValueTree &graphState,
    const LiveGraphCompileOptions &options,
    FrozenBlackBoxResolver resolver) {
  if (!graphState.hasType("GraphDocument"))
    return;
  {
    const juce::ScopedLock lock(requestLock);
    requestedGraph = graphState.createCopy();
    requestedOptions = options;
    requestedResolver = std::move(resolver);
  }
  compilePending.store(true, std::memory_order_release);
  notify();
}

void LiveGraphPublisher::requestRandomization(std::int32_t nodeId,
                                              std::int32_t seed) noexcept {
  {
    const juce::ScopedLock lock(requestLock);
    requestedNodeId = nodeId;
    requestedSeed = seed;
  }
  randomizationPending.store(true, std::memory_order_release);
  notify();
}

std::shared_ptr<LiveGraphRuntime>
LiveGraphPublisher::getPublishedRuntime() const noexcept {
  return std::atomic_load_explicit(&publishedRuntime,
                                   std::memory_order_acquire);
}

juce::String LiveGraphPublisher::getLastError() const {
  const juce::ScopedLock lock(errorLock);
  return lastError;
}

void LiveGraphPublisher::run() {
  while (!threadShouldExit()) {
    wait(-1);
    if (threadShouldExit())
      break;

    while (compilePending.load(std::memory_order_acquire) ||
           randomizationPending.load(std::memory_order_acquire)) {
      if (compilePending.exchange(false, std::memory_order_acq_rel))
        compileLatest();
      if (randomizationPending.exchange(false, std::memory_order_acq_rel))
        randomizeLatest();
      if (threadShouldExit())
        return;
    }
  }
}

void LiveGraphPublisher::compileLatest() {
  juce::ValueTree graphState;
  LiveGraphCompileOptions options;
  FrozenBlackBoxResolver resolver;
  {
    const juce::ScopedLock lock(requestLock);
    graphState = requestedGraph.createCopy();
    options = requestedOptions;
    resolver = requestedResolver;
  }

  graph::NodeGraph graph;
  if (!graph.restoreFromValueTree(graphState)) {
    publishError(
        {LiveGraphErrorCode::invalidGraph, 0,
         "Serialized graph state could not be restored for compilation"});
    return;
  }
  auto compileGraph = graph.withInvisibleCopiesMaterialized();
  const auto result = LiveGraphEngine::compile(compileGraph, options, resolver);
  if (!result.succeeded()) {
    if (result.error.code == LiveGraphErrorCode::incompletePath)
      publishIdle();
    else
      publishError(result.error);
    return;
  }
  LiveGraphCompileError error;
  auto runtime = LiveGraphEngine::prepare(result.snapshot, error);
  if (runtime == nullptr) {
    publishError(error);
    return;
  }
  publish(std::move(runtime));
}

void LiveGraphPublisher::randomizeLatest() {
  std::int32_t nodeId = 0;
  std::int32_t seed = 0;
  {
    const juce::ScopedLock lock(requestLock);
    nodeId = requestedNodeId;
    seed = requestedSeed;
  }

  const auto current = getPublishedRuntime();
  if (current == nullptr) {
    publishError({LiveGraphErrorCode::invalidRandomization, nodeId,
                  "Live graph is not ready for randomization"});
    return;
  }

  LiveGraphCompileError error;
  const auto snapshot =
      current->getSnapshot()->withRandomizedElement(nodeId, seed, error);
  if (snapshot == nullptr) {
    publishError(error);
    return;
  }
  auto runtime = LiveGraphEngine::prepare(snapshot, error);
  if (runtime == nullptr) {
    publishError(error);
    return;
  }
  publish(std::move(runtime));
}

void LiveGraphPublisher::publish(std::shared_ptr<LiveGraphRuntime> runtime) {
  std::atomic_store_explicit(&publishedRuntime, std::move(runtime),
                             std::memory_order_release);
  const juce::ScopedLock lock(errorLock);
  lastError.clear();
}

void LiveGraphPublisher::publishIdle() {
  std::atomic_store_explicit(&publishedRuntime,
                             std::shared_ptr<LiveGraphRuntime>{},
                             std::memory_order_release);
  const juce::ScopedLock lock(errorLock);
  lastError.clear();
}

void LiveGraphPublisher::publishError(const LiveGraphCompileError &error) {
  std::atomic_store_explicit(&publishedRuntime, std::shared_ptr<LiveGraphRuntime>{},
                             std::memory_order_release);
  const juce::ScopedLock lock(errorLock);
  lastError = error.message;
}
} // namespace openyourbox::dsp
