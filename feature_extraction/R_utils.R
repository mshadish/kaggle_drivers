# author: mshadish
##############################
# Utils
#
# This script defines several utility functions for extracting features
# from positional driver data
#
# 1) calcAngle(x,y)
#   - computes the angle given x and y
#
# 2) computeTurn(current angle, previous angle)
#   - computes the turn angle given a current angle/trajectory
#   and a previous angle/trajectory
#   - also accounts for cases in which turns are made from a standstill
#   and whether or not we consider these turns
#
# 3) computeCentripetalAccel(x1,x2,x3,y1,y2,y3,velocity)
#   - computes the centripetal acceleration (v^2/r)
#   given a set of 3 points and the velocity at which the turn is taken
##############################


calcAngle = function(x,y) {
  # computes the true angle, given an x and y
  if (x < 0) {
    atan(y/x) + pi
  }
  else if (x == 0 && y == 0) {
    NaN
  }
  else {
    atan(y/x)
  }
}


computeTurn = function(cur_angle, prev_angle, standstill = FALSE) {
  # computes the turn angle between two trajectory angles
  # check for na's
  if (is.na(cur_angle)) {
    # a) the car is no longer moving, or
    # b) the car is now moving from a standstill
    # in either case, we won't classify this as a turn
    return(0.0)
  }
  else if (!standstill && is.na(prev_angle)) {
    # this indicates that we will not treat moves from a standstill as a turn
    return(0.0)
  }
  else if (standstill && is.na(prev_angle)) {
    # this indicates that we will treat moves from a standstill as a turn
    # this can help us compute overall trajectory
    prev_angle = 0.0
  }
  
  # run a while loop to minimize the turn angle
  # to prevent the creation of extreme turns that can be represented
  # as more realistic turns in the opposite direction
  # e.g. a 340 degree right turn (which is a 20 deg left turn)
  turn = cur_angle - prev_angle
  while (abs(turn) > abs(turn - sign(turn) * 2 * pi)) {
    turn = turn - sign(turn) * 2 * pi
    # write an if for the rare case we have an exactly 180 degree turn
    # which probably is incorrect, but we want to avoid the potential for an infinite loop
    if (abs(turn) == pi) {
      break
    }
  }
  return(turn)
}



computeCentripetalAccel = function(x_1, x_2, x_3, y_1, y_2, y_3, apex_velocity) {
  # computes the centripetal acceleration when given 3 consecutive points
  # this is achieved by
  #   1) computing the center of the circle that passes through all three points, then...
  #   2) dividing the square velocity of the car at the center of the 'turn' (middle point)
  #   by the computed radius
  # Note that we must pass in apex velocity in case we must re-order our points
  # when computing the slopes
  
  # first, compute the slopes
  slope_1 = (y_2 - y_1) / (x_2 - x_1)
  slope_2 = (y_3 - y_2) / (x_3 - x_2)
  
  # an NA slope implies 0/0, suggesting that the car did not move
  # at one of the three points
  # meaning no centripetal acceleration was seen
  if (is.na(slope_1) || is.na(slope_2)) {
    return(0.0)
  }
  
  # if they are equivalent, no turn is occurring
  # therefore, our centripetal acceleration is 0
  else if (slope_1 == slope_2) {
    return(0.0)
  }
  
  # if they are different, then a turn is occurring
  # however, if one of the slopes is infinite, we must compensate
  # and compute the turn radius based on two other pairs of points
  else if (slope_1 == Inf || slope_2 == Inf) {
    # we can achieve this simply by recursively calling our function,
    # shifting our points one
    return(computeCentripetalAccel(x_2, x_3, x_1, y_2, y_3, y_1, apex_velocity))
  }
  
  # otherwise, we can continue with the computation as normal
  #######################################################
  # we must compute the location of the center of the circle
  # by first finding the x-coordinate
  numerator = (slope_1*slope_2*(y_1-y_3)) + (slope_2*(x_1+x_2)) - (slope_1*(x_2+x_3))
  denominator = 2 * (slope_2 - slope_1)
  center_x = numerator / denominator
  # next, we apply this x-coordinate to find the corresponding y-coordinate
  center_y = -((1/slope_1) * (center_x - (x_1 + x_2)/2)) + (y_1 + y_2)/2
  
  # to compute the radius length, we simply compute the distance
  # between our circle center and any arbitrary point
  radius = sqrt((center_x - x_1)^2 + (center_y - y_1)^2)
  
  #######################################################
  # with our radius, we can now compute centripetal acceleration
  centripetal_accel = apex_velocity^2 / radius
  return(centripetal_accel)
}