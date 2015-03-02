# author: mshadish
########################################
# This script is the main function
# to extract features from the kaggle drivers dataset
########################################
########################################
# DEFINE THE PATH TO THE FOLDERS HERE
path = 'drivers'
########################################
# source the utility function file
source('R_utils.R')
options(warn = -1)


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
  
  avg_right_turn_centripetal_accel = rep(NA, 200)
  max_right_turn_centripetal_accel = rep(NA, 200)
  avg_left_turn_centripetal_accel = rep(NA, 200)
  max_left_turn_centripetal_accel = rep(NA, 200)
  

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
    # pull out the x's and y's
    x_s = data$x
    y_s = data$y
    # difference it to obtain velocities
    diffed_x = diff(x_s)
    diffed_y = diff(y_s)

    ########################################################
    # FIRST, WE WILL CALCULATE THE TURNS

    # calculate the angles
    angles = mapply(FUN = calcAngle, diffed_x, diffed_y)
    lag_angles = angles[-length(angles)]
    current_angles = angles[-1]
    # now calculate the turns
    turn = mapply(FUN = computeTurn, current_angles, lag_angles)
    # in this case, positive is left and negative is right
    # note that moving from a standstill does not constitute a turn

    # TURNING FEATURES
    left_turns = subset(turn, turn > 0)
    right_turns = subset(turn, turn < 0)
    left_turn_frac = length(left_turns) / length(turn)
    avg_left_turn = mean(left_turns)
    med_left_turn = median(left_turns)
    max_left_turn = max(left_turns, na.rm = TRUE)
    sd_left_turn = sd(left_turns)
    right_turn_frac = length(right_turns) / length(turn)
    avg_right_turn = mean(right_turns)
    med_right_turn = median(right_turns)
    max_right_turn = min(right_turns, na.rm = TRUE)
    sd_right_turn = sd(right_turns)

    # while we're at it, let's compute the trajectory of the vehicle
    trajectory = mapply(FUN = computeTurn, current_angles, lag_angles,
                        rep(TRUE, length(current_angles)))
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
    max_vel = max(vel, na.rm = TRUE)
    time_cruising = length(vel_no_0) / length(vel)
    distance = sum(vel)
    

    ########################################################
    # CALCULATE THE ACCELERATION METRICS
    accel = diff(vel)
    accel_no_0 = subset(accel, accel != 0)
    accel_pos = subset(accel, accel > 0)
    accel_neg = subset(accel, accel < 0)
    # acceleration features
    max_accel = max(accel, na.rm = TRUE)
    max_brake = min(accel, na.rm = TRUE)
    time_accel = length(accel_pos) / length(accel)
    time_braking = length(accel_neg) / length(accel)
    
    avg_accel = mean(accel_pos)
    med_accel = median(accel_pos)
    # in case we have no acceleration (aka no movement)
    if (length(accel_pos) == 0) {
      avg_accel = 0
      med_accel = 0
    }
    
    # in case we happen to have no braking, we want to capture that
    avg_braking = mean(accel_neg)
    med_braking = median(accel_neg)
    if (length(accel_neg) == 0) {
      avg_braking = 0
      med_braking = 0
    }
    
    
    ########################################################
    # HERE WE WILL CALCULATE THE CENTRIPETAL ACCELERATION EXPERIENCED AT EACH POINT
    # note that each computation will require a lag, current, and leading position
    
    # drop the last 2 vars in each list to get the lag position
    lag_x = x_s[-c(length(x_s) - 1, length(x_s))]
    lag_y = y_s[-c(length(y_s) - 1, length(y_s))]
    # drop the first and last vars to get the current position
    current_x = x_s[-c(1, length(x_s))]
    current_y = y_s[-c(1, length(y_s))]
    # drop the first 2 vars to get the lead position
    lead_x = x_s[-c(1,2)]
    lead_y = y_s[-c(1,2)]

    # to compute the apex velocities, we average the velocity before and after
    # each turn apex (a.k.a. our 'current' position)
    apex_velocities = mapply(function(x,y) {(x+y)/2}, vel[-1], vel[-length(vel)])
    # with our apex velocities, we can compute our centripetal accelerations
    centripetal_accel = mapply(computeCentripetalAccel,
                               lag_x, current_x, lead_x, lag_y, current_y, lead_y,
                               apex_velocities)
    # let's break this out by left and right turn
    left_turn_indices = as.logical(turn > 0)
    centripetal_accel_left = centripetal_accel[left_turn_indices]
    right_turn_indices = as.logical(turn < 0)
    centripetal_accel_right = centripetal_accel[right_turn_indices]
    # in both cases, let's filter out 0's that may have slipped through the cracks
    centripetal_accel_left = subset(centripetal_accel_left, centripetal_accel_left > 0)
    centripetal_accel_right = subset(centripetal_accel_right, centripetal_accel_right > 0)
    # now we must boil these down into single features
    avg_centripetal_accel_left = mean(centripetal_accel_left, na.rm = TRUE)
    max_centripetal_accel_left = max(centripetal_accel_left, na.rm = TRUE)
    avg_centripetal_accel_right = mean(centripetal_accel_right, na.rm = TRUE)
    max_centripetal_accel_right = max(centripetal_accel_right, na.rm = TRUE)

    
    ########################################################
    # NOW ADD THE FEATURES TO THE LISTS
    # note: we use is.null() statements to catch null's
    # and replace them with 0's
    left_turns_taken[counter] = length(left_turns)
    right_turns_taken[counter] = length(right_turns)
    left_turn_fraction[counter] = left_turn_frac
    avg_left_turn_angle[counter] = avg_left_turn
    med_left_turn_angle[counter] = ifelse(!is.null(med_left_turn), med_left_turn, 0)
    max_left_turn_angle[counter] = ifelse(!(max_left_turn == -Inf), max_left_turn, 0)
    sd_left_turn_angle[counter] = sd_left_turn
    right_turn_fraction[counter] = right_turn_frac
    avg_right_turn_angle[counter] = avg_right_turn
    med_right_turn_angle[counter] = ifelse(!is.null(med_right_turn), med_right_turn, 0)
    max_right_turn_angle[counter] = ifelse(!(max_right_turn == -Inf), max_right_turn, 0)
    sd_right_turn_angle[counter] = sd_right_turn

    final_direction[counter] = final_dir

    avg_velocity[counter] = avg_vel
    med_velocity[counter] = ifelse(!is.null(med_vel), med_vel, 0)
    avg_velocity_no_0[counter] = avg_vel_no_0
    med_velocity_no_0[counter] = ifelse(!is.null(med_vel_no_0), med_vel_no_0, 0)
    max_velocity[counter] = ifelse(!(max_vel == -Inf), max_vel, 0)
    time_spent_cruising[counter] = time_cruising
    distance_traveled[counter] = distance

    max_acceleration[counter] = ifelse(!(max_accel == -Inf), max_accel, 0)
    max_deceleration[counter] = ifelse(!(max_brake == -Inf), max_brake, 0)
    time_spent_accelerating[counter] = time_accel
    time_spent_braking[counter] = time_braking
    avg_acceleration[counter] = avg_accel
    med_acceleration[counter] = ifelse(!is.null(med_accel), med_accel, 0)
    avg_deceleration[counter] = avg_braking
    med_deceleration[counter] = ifelse(!is.null(med_braking), med_braking, 0)
    
    avg_right_turn_centripetal_accel[counter] = ifelse(!is.na(avg_centripetal_accel_right),
                                                       avg_centripetal_accel_right, 0)
    max_right_turn_centripetal_accel[counter] = ifelse(!(max_centripetal_accel_right == -Inf),
                                                       max_centripetal_accel_right, 0)
    avg_left_turn_centripetal_accel[counter] = ifelse(!is.na(avg_centripetal_accel_left),
                                                      avg_centripetal_accel_left, 0)
    max_left_turn_centripetal_accel[counter] = ifelse(!(max_centripetal_accel_left == -Inf),
                                                      max_centripetal_accel_left, 0)

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
                         med_deceleration, avg_right_turn_centripetal_accel,
                         max_right_turn_centripetal_accel, avg_left_turn_centripetal_accel,
                         max_left_turn_centripetal_accel)
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
folders = sapply(dir(path), FUN = function(x) {paste(path, x, sep = '/')})
# and create the individual dataframe csv's
sapply(folders, FUN = createIndividualDriverDf)