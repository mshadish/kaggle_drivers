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
    left_turns_l[counter] = length(left_turns)
    right_turns_l[counter] = length(right_turns)
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
    counter = counter + 1
    #print(paste('Completed', toString(counter), sep=' '))
  }
  # now that all of the vectors have been populated,
  # create the returning data frame
  return_df = data.frame(id_list, left_turns_l, right_turns_l, left_turn_frac_l,
                         avg_left_turn_l, med_left_turn_l, max_left_turn_l,
                         sd_left_turn_l, right_turn_frac_l, avg_right_turn_l,
                         med_right_turn_l, max_right_turn_l, sd_right_turn_l,
                         final_dir_l, avg_vel_l, med_vel_l, avg_vel_no_0_l,
                         med_vel_no_0_l, max_vel_l, time_cruising_l,
                         distance_l, max_accel_l, max_brake_l, time_accel_l,
                         time_braking_l, avg_accel_l, med_accel_l, avg_braking_l,
                         med_braking_l)
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
folders = sapply(dir('drivers/drivers'), FUN = function(x) {paste('drivers/drivers/', x, sep = '')})
# and create the individual dataframe csv's
sapply(folders, FUN = createIndividualDriverDf)