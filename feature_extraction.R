# author: mshadish
########################################
# This script defines several functions used
# to extract features from the driver folders
########################################

calcAngle = function(x,y) {
  # function to compute the true angle
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
  # function to compute the turn between two trajectory angles
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
  
  comparison = cur_angle - (sign(cur_angle) * 2 * pi)
  # compare absolute values
  if (abs(prev_angle - cur_angle) > abs(prev_angle - comparison)) {
    # in this case, the converted angle provides the minimum
    comparison - prev_angle
  }
  else {
    # otherwise, keep the angle
    cur_angle - prev_angle
  }
}


createIndividualDriverDf = function(folder_name) {
  # this function creates a data frame representing all of the files
  # within a given driver folder

  # initialize lists to store results
  left_turns_l = rep(NA, 200)
  right_turns_l = rep(NA, 200)
  left_turn_frac_l = rep(NA, 200)
  avg_left_turn_l = rep(NA, 200)
  med_left_turn_l = rep(NA, 200)
  max_left_turn_l = rep(NA, 200)
  sd_left_turn_l = rep(NA, 200)
  right_turn_frac_l = rep(NA, 200)
  avg_right_turn_l = rep(NA, 200)
  med_right_turn_l = rep(NA, 200)
  max_right_turn_l = rep(NA, 200)
  sd_right_turn_l = rep(NA, 200)

  final_dir_l = rep(NA, 200)

  avg_vel_l = rep(NA, 200)
  med_vel_l = rep(NA, 200)
  avg_vel_no_0_l = rep(NA, 200)
  med_vel_no_0_l = rep(NA, 200)
  max_vel_l = rep(NA, 200)
  time_cruising_l = rep(NA, 200)
  distance_l = rep(NA, 200)

  max_accel_l = rep(NA, 200)
  max_brake_l = rep(NA, 200)
  time_accel_l = rep(NA, 200)
  time_braking_l = rep(NA, 200)
  avg_accel_l = rep(NA, 200)
  med_accel_l = rep(NA, 200)
  avg_braking_l = rep(NA, 200)
  med_braking_l = rep(NA, 200)

  # set up files to loop through
  files = list.files(folder_name, pattern = '*.csv')

  # now loop through all of the files
  counter = 1
  for (file in files) {

    # read in the data
    data = read.csv(paste(folder_name, file, sep = '/'))
    # difference it to obtain velocities
    diffed_x = diff(data$x)
    diffed_y = diff(data$y)

    ########################################################
    # FIRST, WE WILL CALCULATE THE TURNS

    # calculate the angles
    angles = mapply(FUN = calcAngle, diffed_x, diffed_y)
    lag_angles = c(0, angles[-length(angles)])
    # now calculate the turns
    turn = mapply(FUN = computeTurn, angles, lag_angles)
    # in this case, positive is left and negative is right
    # note that moving from a standstill does not constitute a turn

    # TURNING FEATURES
    left_turns = subset(turn, turn > 0)
    right_turns = subset(turn, turn < 0)
    left_turn_frac = length(left_turns) / length(turn)
    avg_left_turn = mean(left_turns)
    med_left_turn = median(left_turns)
    max_left_turn = max(left_turns)
    sd_left_turn = sd(left_turns)
    right_turn_frac = length(right_turns) / length(turn)
    avg_right_turn = mean(right_turns)
    med_right_turn = median(right_turns)
    max_right_turn = min(right_turns)
    sd_right_turn = sd(right_turns)

    # while we're at it, let's compute the trajectory of the vehicle
    trajectory = mapply(FUN = computeTurn, angles, lag_angles,
                        rep(TRUE, length(angles)))
    # again, positive is left and negative is right
    # TRAJECTORY FEATURES
    final_dir = sum(trajectory)
    # normalize final direction
    while (final_dir < 0) {
      final_dir = final_dir + (2*pi)
    }
    final_dir = final_dir %% (2*pi)


    ########################################################
    # NEXT, WE WILL CALCULATE THE VELOCITY METRICS
    vel = sqrt(diffed_x^2 + diffed_y^2)
    vel_no_0 = subset(vel, vel > 0)
    # velocity features
    avg_vel = mean(vel)
    med_vel = median(vel)
    avg_vel_no_0 = mean(vel_no_0)
    med_vel_no_0 = median(vel_no_0)
    max_vel = max(vel)
    time_cruising = length(vel_no_0) / length(vel)
    distance = sum(vel)


    ########################################################
    # FINALLY, WE WILL CALCULATE THE ACCURACY METRICS
    accel = diff(vel)
    accel_no_0 = subset(accel, accel != 0)
    accel_pos = subset(accel, accel > 0)
    accel_neg = subset(accel, accel < 0)
    # acceleration features
    max_accel = max(accel)
    max_brake = min(accel)
    time_accel = length(accel_pos) / length(accel)
    time_braking = length(accel_neg) / length(accel)
    avg_accel = mean(accel_pos)
    med_accel = median(accel_pos)
    avg_braking = mean(accel_neg)
    med_braking = median(accel_neg)


    ########################################################
    # NOW ADD THE FEATURES TO THE LISTS
    left_turns_l[counter] = left_turns
    right_turns_l[counter] = right_turns
    left_turn_frac_l[counter] = left_turn_frac
    avg_left_turn_l[counter] = avg_left_turn
    med_left_turn_l[counter] = med_left_turn
    max_left_turn_l[counter] = max_left_turn
    sd_left_turn_l[counter] = sd_left_turn
    right_turn_frac_l[counter] = right_turn_frac
    avg_right_turn_l[counter] = avg_right_turn
    med_right_turn_l[counter] = med_right_turn
    max_right_turn_l[counter] = max_right_turn
    sd_right_turn_l[counter] = sd_right_turn

    final_dir_l[counter] = final_dir

    avg_vel_l[counter] = avg_vel
    med_vel_l[counter] = med_vel
    avg_vel_no_0_l[counter] = avg_vel_no_0
    med_vel_no_0_l[counter] = med_vel_no_0
    max_vel_l[counter] = max_vel
    time_cruising_l[counter] = time_cruising
    distance_l[counter] = distance

    max_accel_l[counter] = max_accel
    max_brake_l[counter] = max_brake
    time_accel_l[counter] = time_accel
    time_braking_l[counter] = time_braking
    avg_accel_l[counter] = avg_accel
    med_accel_l[counter] = med_accel
    avg_braking_l[counter] = avg_braking
    med_braking_l[counter] = med_braking

    # increment the counter
    print(paste('Completed', toString(counter), sep=' '))
    counter = counter + 1
  }
  return
}