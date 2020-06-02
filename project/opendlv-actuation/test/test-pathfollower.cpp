/*
 * Copyright (C) 2018 Ola Benderius
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <cmath>

#include "cluon-complete.hpp"
#include "opendlv-extended-message-set.hpp"

#include "pathfollower.hpp"


TEST_CASE("Test pathFollower: Stops driving if front distance is short") {
  PathFollower pathfollower;

  opendlv::proxy::DistanceReading distanceReading;
  distanceReading.distance(0.05f);
  pathfollower.setFrontUltrasonic(distanceReading);

  pathfollower.step();
  auto pp = pathfollower.getPedalPositionRequest();

  REQUIRE(pp.position() == Approx(0.0f));
}

TEST_CASE("Test pathFollower: Turns on cones detected") {
  PathFollower pathfollower;
  opendlv::logic::perception::Cones conesReading;

  opendlv::proxy::DistanceReading distanceReading;
  distanceReading.distance(0.5f);
  pathfollower.setFrontUltrasonic(distanceReading);

  SECTION("Drives forward") {
    pathfollower.step();
    auto pp = pathfollower.getPedalPositionRequest();
    REQUIRE(pp.position() > 0.0f);
  }

  SECTION("Turn left if we only observe cones to the right") {
    conesReading.color(0); // Blue
    conesReading.i("400,200");
    conesReading.j("370,100");

    pathfollower.setConesPixelPosition(conesReading);
    pathfollower.step();

    auto gsa = pathfollower.getGroundSteeringAngle();

    REQUIRE(gsa.groundSteering() > 0);
  }

  SECTION("Turn right if we only observe cones to the left") {
    conesReading.color(1); // Yellow
    conesReading.i("-400,-200");
    conesReading.j("370,100");

    pathfollower.setConesPixelPosition(conesReading);
    pathfollower.step();

    auto gsa = pathfollower.getGroundSteeringAngle();

    REQUIRE(gsa.groundSteering() < 0);
  }
}

// TEST_CASE("Test pathFollower, a short distance to the front should make Kiwi stop.") {
//   PathFollower pathfollower;

//   opendlv::proxy::DistanceReading distanceReading;
//   distanceReading.distance(0.1f);
  
//   pathfollower.setFrontUltrasonic(distanceReading);
//   pathfollower.step();

//   auto pp = pathfollower.getPedalPositionRequest();

//   REQUIRE(pp.position() == Approx(0.0f));
// }
