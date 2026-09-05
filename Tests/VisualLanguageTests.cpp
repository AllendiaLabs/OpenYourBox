#include "ui/VisualLanguage.h"

#include <cstdio>
#include <cstdlib>

namespace {
using openyourbox::ui::Rgba;
using openyourbox::ui::VisualLanguage;

bool expect(bool condition, const char *message) {
  if (!condition)
    std::fprintf(stderr, "VisualLanguageTests: %s\n", message);
  return condition;
}

bool pairwiseDistinct(const Rgba &a, const Rgba &b, const char *pair) {
  char buffer[128];
  std::snprintf(buffer, sizeof buffer, "%s must be arm's-length distinct", pair);
  return expect(VisualLanguage::rgbDistance(a, b) >= 0.25f, buffer);
}
} // namespace

int main() {
  bool passed = true;
  passed &= expect(VisualLanguage::isDarkSurface(VisualLanguage::Surface::page),
                   "page must be a dark surface");
  passed &= expect(VisualLanguage::isDarkSurface(VisualLanguage::Surface::canvas),
                   "canvas must be a dark surface");
  passed &= expect(VisualLanguage::isDarkSurface(VisualLanguage::Surface::panel),
                   "panel must be a dark surface");
  passed &= expect(VisualLanguage::isDarkSurface(VisualLanguage::Surface::raised),
                   "raised must be a dark surface");
  passed &= expect(VisualLanguage::relativeLuminance(VisualLanguage::Surface::page) <
                       VisualLanguage::relativeLuminance(
                           VisualLanguage::Surface::canvas),
                   "canvas must lift above page");
  passed &= pairwiseDistinct(VisualLanguage::live, VisualLanguage::frozen,
                             "live vs frozen");
  passed &= pairwiseDistinct(VisualLanguage::live, VisualLanguage::accent,
                             "live vs accent");
  passed &= pairwiseDistinct(VisualLanguage::frozen, VisualLanguage::accent,
                             "frozen vs accent");
  passed &= pairwiseDistinct(VisualLanguage::live, VisualLanguage::danger,
                             "live vs danger");
  passed &= pairwiseDistinct(VisualLanguage::frozen, VisualLanguage::danger,
                             "frozen vs danger");
  passed &= pairwiseDistinct(VisualLanguage::accent, VisualLanguage::danger,
                             "accent vs danger");
  passed &= pairwiseDistinct(VisualLanguage::Family::audioIo,
                             VisualLanguage::Family::conditioning,
                             "audioIo vs conditioning");
  passed &= pairwiseDistinct(VisualLanguage::Family::audioIo,
                             VisualLanguage::Family::helper, "audioIo vs helper");
  passed &= pairwiseDistinct(VisualLanguage::Family::audioIo,
                             VisualLanguage::Family::trainOnly,
                             "audioIo vs trainOnly");
  passed &= pairwiseDistinct(VisualLanguage::Family::conditioning,
                             VisualLanguage::Family::helper,
                             "conditioning vs helper");
  passed &= pairwiseDistinct(VisualLanguage::Family::conditioning,
                             VisualLanguage::Family::trainOnly,
                             "conditioning vs trainOnly");
  passed &= pairwiseDistinct(VisualLanguage::Family::helper,
                             VisualLanguage::Family::trainOnly,
                             "helper vs trainOnly");
  passed &= pairwiseDistinct(VisualLanguage::Family::audioIo, VisualLanguage::live,
                             "audioIo vs live");
  passed &= pairwiseDistinct(VisualLanguage::Family::helper, VisualLanguage::frozen,
                             "helper vs frozen");
  passed &= expect(VisualLanguage::Type::bodyWeight == 400 &&
                       VisualLanguage::Type::strongWeight == 600,
                   "Inter ramp must register Regular 400 and SemiBold 600");
  passed &= expect(VisualLanguage::Type::bodyWeight !=
                       VisualLanguage::Type::strongWeight,
                   "exactly two Inter weights");
  if (!passed)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
