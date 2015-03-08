# author: mshadish
################################################################################
# This script contains some of the EDA performed
# on the features extracted from the driver telematics data
################################################################################
# NOTE: WE WILL DEFINE OUR CONSTANT DIRECTORY HERE FOR USABILITY
dir_of_files = '/Users/mshadish/git_repos/kaggle_drivers/extracted/'
################################################################################
library(ggplot2)
library(reshape2)
options(warn = -1)

# WE WILL DEFINE SEVERAL FUNCTIONS THAT, IN CONJUNCTION WITH EACH OTHER,
# WILL PLOT SIDE-BY-SIDE BOXPLOTS OF OUR DATA FOR COMPARISON
createDFForBoxplot = function(list_numbers, col_name, directory = '.',
                              suffix = '_summary.csv') {
  #######################
  # THIS FUNCTION CREATES A DATAFRAME WITH COLUMNS CORRESPONDING TO
  # THE SPECIFIED COLUMN NAME FROM A LIST OF FILES
  #######################
  
  # create a dataframe to store our boxplot data
  # initialized with 200 rows
  df_store = data.frame(n = c(1:200))
  # step through the numbers in the list passed in
  for (num in list_numbers) {
    # generate a file name
    file_name = paste(num, suffix, sep = '')
    # and test to verify it is in the given directory
    if (!file_name %in% list.files(directory)) {
      print(paste('File', num,'not found', sep = ' '))
      next
    }
    # otherwise, we are free to open the file
    data = read.csv(paste(directory, file_name, sep = '/'))
    # and keep track of the record in our dataframe
    df_store[[strsplit(file_name, '_')[[1]][1]]] = data[[col_name]]
  }
  # remove our initializing column
  df_store = subset(df_store, select = -n)
  
}


genBoxplot = function(list_numbers, col_name, clean_name = toupper(sub('_', ' ', col_name)),
                      directory = '.', suffix = '_summary.csv') {
  #######################
  # THIS FUNCTION PLOTS THE BOXPLOTS FOR A GIVEN COLUMN NAME FROM GIVEN FILES
  #######################
  
  # pull out the data
  df_store = createDFForBoxplot(list_numbers, col_name, directory, suffix)
  
  # let's compute the boxplot stats for each column
  # with the intention of only displaying records within the records
  # to improve readability
  # this will be achieved by computing the minimum and maximum whisker values
  y_min = Inf
  y_max = -Inf
  # loop through the columns in the data frame
  for (file in df_store) {
    # compute the boxplot whiskers
    curr_y_low = boxplot.stats(file)$stats[1]
    curr_y_high = boxplot.stats(file)$stats[5]
    # update our min and max
    if (curr_y_low < y_min) {
      y_min = curr_y_low
    }
    if (curr_y_high > y_max) {
      y_max = curr_y_high
    }
  }
  # let's adjust the min and max such that the plot doesn't cut into the whisker
  y_range = abs(y_min - y_max)
  y_min = y_min - y_range * 0.05
  y_max = y_max + y_range * 0.05
  
  ##########################
  # PLOTTING THE DATA
  ##########################
  # melt the df
  melted = melt(df_store, measure.vars = seq(1,length(df_store)))
  # and now we can plot using ggplot
  plot_obj = ggplot(melted)
  plot_obj = plot_obj + geom_boxplot(aes(x = variable, y = value, color = variable),
                                     outlier.shape = NA)
  # modify the axes
  plot_obj = plot_obj + scale_x_discrete(breaks = NULL) + ylab(clean_name) + xlab('Driver')
  plot_obj = plot_obj + coord_cartesian(ylim = c(y_min, y_max))
  # clean up the legend
  plot_obj = plot_obj + labs(color = 'Driver')
  # and add a title
  plot_obj = plot_obj + ggtitle(paste('Boxplots of', clean_name, sep = ' '))
  
  # display the object
  print(plot_obj)
  # and save
  ggsave(filename = paste(col_name, 'png', sep = '.'), plot = plot_obj)
}

##########################################
# let's generate a random sample of 10 files
rand_files = sample(1:3612, size = 5)
# note: the file naming scheme is strange,
# so it's normal to not find some files between 1 and 3612
# the function will take care of this
##########################################
# LEFT TURN FRACTION

##########################################
# AVG LEFT TURN ANGLE

##########################################
# MEDIAN LEFT TURN ANGLE

##########################################
# MAXIMUM LEFT TURN ANGLE

##########################################
# SD OF LEFT TURN ANGLES

##########################################
# RIGHT TURN FRACTION

##########################################
# AVERAGE RIGHT TURN ANGLE

##########################################
# MEDIAN RIGHT TURN ANGLE

##########################################
# MAXIMUM RIGHT TURN ANGLE

##########################################
# SD OF RIGHT TURN ANGLES

##########################################
# FINAL DIRECTION

##########################################
# AVERAGE VELOCITY

##########################################
# MEDIAN VELOCITY

##########################################
# AVERAGE VELOCITY, WITH 0'S REMOVED

##########################################
# MEDIAN VELOCITY, 0'S REMOVED

##########################################
# MAXIMUM VELOCITY ACHIEVED

##########################################
# TIME SPENT CRUISING

##########################################
# TOTAL DISTANCE TRAVELED
# distance traveled shows some decent variation in the distributions

##########################################
# MAXIMUM ACCELERATION ACHIEVED

##########################################
# MAXIMUM BRAKING

##########################################
# TIME SPENT ACCELERATING

##########################################
# TIME SPENT BRAKING

##########################################
# AVERAGE ACCELERATION
genBoxplot(rand_files, 'avg_acceleration', 'Average Acceleration',
           directory = dir_of_files)
# average acceleration appears heavily skewed regardless

##########################################
# MEDIAN ACCELERATION
# med acceleration shows mild variation in distributions

##########################################
# AVERAGE DECELERATION/BRAKING

##########################################
# MEDIAN DECELERATION/BRAKING
# median braking appears to demonstrate different distributions

##########################################
# AVERAGE RIGHT TURN CENTRIPETAL ACCELERATION
genBoxplot(rand_files, 'avg_right_turn_centripetal_accel',
           'Average Right Turn Centripetal Acceleration', directory = dir_of_files)


##########################################
# MAX RIGHT TURN CENTRIPETAL ACCELERATION
genBoxplot(rand_files, 'max_right_turn_centripetal_accel',
           'Maximum Right Turn Centripetal Acceleration', directory = dir_of_files)