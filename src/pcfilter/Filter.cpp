// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)#pragma once
#include "src/pcfilter/Filter.h"

int Filter::instanceCounter = 0;
std::vector<FilterFactory*> FilterFactory::availableFilterFactories = std::vector<FilterFactory*>();
