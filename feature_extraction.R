# author: mshadish
########################################
# This script defines several functions used
# to extract features from the driver folders
########################################
options(warn = -1)

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
  
  # run a while loop to minimize the turn angle
  # to prevent the creation of a 340 degree right turn (which is a 20 deg left turn)
  turn = cur_angle - prev_angle
  while (abs(turn) > abs(turn - sign(turn) * 2 * pi)) {
    turn = turn - sign(turn) * 2 * pi
    # write an if for the rare case we have an exactly 90 degree turn
    if (abs(turn) == pi/2) {
      break
    }
  }
  # return
  turn
  
  
#   comparison = cur_angle - (sign(cur_angle) * 2 * pi)
#   # compare absolute values
#   if (abs(prev_angle - cur_angle) > abs(prev_angle - comparison)) {
#     # in this case, the converted angle provides the minimum
#     comparison - prev_angle
#   }
#   else {
#     # otherwise, keep the angle
#     cur_angle - prev_angle
#   }
}


createIndividualDriverDf = function(folder_dir) {
  # this function creates a data frame representing all of the files
  # within a given driver folder
  # note that, instead of continually appending to our vectors
  # (which would be slow)
  # we will first initialize our lists of length 200
  # and simply iterate a positioning counter to determine
  # where to insert the next element within the 200
  
  # first, pull out the folder name from the given directory
  directory_split = strsplit(folder_dir, split = '/')[[1]]
  folder_name = directory_split[length(directory_split)]

  # initialize lists to store results
  # we know that every folder contains 200 driver files
  id_list = rep(NA, 200)
  left_turns_taken = rep(NA, 200)
  right_turns_taken = rep(NA, 200)
  left_turn_fraction = rep(NA, 200)
  avg_left_turn_angle = rep(NA, 200)
  med_left_turn_angle = rep(NA, 200)
  max_left_turn_angle = rep(NA, 200)
  sd_left_turn_angle = rep(NA, 200)
  right_turn_fraction = rep(NA, 200)
  avg_right_turn_angle = rep(NA, 200)
  med_right_turn_angle = rep(NA, 200)
  max_right_turn_angle = rep(NA, 200)
  sd_right_turn_angle = rep(NA, 200)

  final_direction = rep(NA, 200)

  avg_velocity = rep(NA, 200)
  med_velocity = rep(NA, 200)
  avg_velocity_no_0 = rep(NA, 200)
  med_velocity_no_0 = rep(NA, 200)
  max_velocity = rep(NA, 200)
  time_spent_cruising = rep(NA, 200)
  distance_traveled = rep(NA, 200)

  max_acceleration = rep(NA, 200)
  max_deceleration = rep(NA, 200)
  time_spent_accelerating = rep(NA, 200)
  time_spent_braking = rep(NA, 200)
  avg_acceleration = rep(NA, 200)
  med_acceleration = rep(NA, 200)
  avg_deceleration = rep(NA, 200)
  med_deceleration = rep(NA, 200)

  # set up files to loop through
  files = list.files(folder_dir, pattern = '*.csv')

  # now loop through all of the files
  counter = 1
  for (file in files) {
    
    # grab the ID of this file (stripping out the '.csv')
    file_id = strsplit(file, split = '.', fixed = TRUE)[[1]][1]
    folder_file_id = paste(folder_name, file_id, sep = '_')
    id_list[counter] = folder_file_id

    # read in the data
    data = read.csv(paste(folder_dir, file, sep = '/'))
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
    left_turns_taken[counter] = length(left_turns)
    right_turns_taken[counter] = length(right_turns)
    left_turn_fraction[counter] = left_turn_frac
    avg_left_turn_angle[counter] = avg_left_turn
    med_left_turn_angle[counter] = med_left_turn
    max_left_turn_angle[counter] = max_left_turn
    sd_left_turn_angle[counter] = sd_left_turn
    right_turn_fraction[counter] = right_turn_frac
    avg_right_turn_angle[counter] = avg_right_turn
    med_right_turn_angle[counter] = med_right_turn
    max_right_turn_angle[counter] = max_right_turn
    sd_right_turn_angle[counter] = sd_right_turn

    final_direction[counter] = final_dir

    avg_velocity[counter] = avg_vel
    med_velocity[counter] = med_vel
    avg_velocity_no_0[counter] = avg_vel_no_0
    med_velocity_no_0[counter] = med_vel_no_0
    max_velocity[counter] = max_vel
    time_spent_cruising[counter] = time_cruising
    distance_traveled[counter] = distance

    max_acceleration[counter] = max_accel
    max_deceleration[counter] = max_brake
    time_spent_accelerating[counter] = time_accel
    time_spent_braking[counter] = time_braking
    avg_acceleration[counter] = avg_accel
    med_acceleration[counter] = med_accel
    avg_deceleration[counter] = avg_braking
    med_deceleration[counter] = med_braking

    # increment the counter
    counter = counter + 1
    #print(paste('Completed', toString(counter), sep=' '))
  }
  # now that all of the vectors have been populated,
  # create the returning data frame
  return_df = data.frame(id_list, left_turns_taken, right_turns_taken, left_turn_fraction,
                         avg_left_turn_angle, med_left_turn_angle, max_left_turn_angle,
                         sd_left_turn_angle, right_turn_fraction, avg_right_turn_angle,
                         med_right_turn_angle, max_right_turn_angle, sd_right_turn_angle,
                         final_direction, avg_velocity, med_velocity, avg_velocity_no_0,
                         med_velocity_no_0, max_velocity, time_spent_cruising,
                         distance_traveled, max_acceleration, max_deceleration, time_spent_accelerating,
                         time_spent_braking, avg_acceleration, med_acceleration, avg_deceleration,
                         med_deceleration)
  # report completion
  print(paste('Completed folder', folder_dir, sep = ' '))
  # write to a csv
  write.table(return_df, file = paste(folder_dir, '_summary.csv', sep = ''),
              sep = ',', quote = FALSE, row.names = FALSE)
  # end of function
}


############################
# MAIN FUNCTION
############################
# grab the names of folders
folders = sapply(dir('drivers'), FUN = function(x) {paste('drivers/', x, sep = '')})
# and create the individual dataframe csv's
sapply(folders, FUN = createIndividualDriverDf)