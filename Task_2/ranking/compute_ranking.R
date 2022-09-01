#!/usr/bin/env Rscript

library(yaml)
library(dplyr)
library(reshape2)
library(challengeR)
library(doParallel)
library(huxtable)
library(magrittr)

# Function to calculate one subranking ------------------------------------

#' Calculate a single ranking
#'
#' @param data The underlying dataset used to calculate the ranking (data.frame)
#' @param metric_variant Either "Dice" or "Hausdorff95" (str)
#' @param institution_name Name of the institution (used in title) (str)
#' @param ranking_method Ranking method, choose from
#' (rankThenMean, rankThenMedian, aggregateThenMean, aggregateThenMedian, testBased)
#' @param title_name_ending Ending of the title string (str)
#' @param file_name_ending Ending of the file name string (str)
#'
#' @return A ranking list

calculate_sub_ranking <- function(data, metric_variant, institution_name, ranking_method,
                                  title_name_ending, file_name_ending, report_dir = NULL) {

  smallBetter_order <- FALSE
  isna <- 0
  if(metric_variant == "Hausdorff95") {
    smallBetter_order <- TRUE
    isna <- 1000
  }

  if(sum(is.na(data$metric_value))>0) {
    challenge <- as.challenge(data, algorithm = "algorithm", case = "case", value = "metric_value",
                              smallBetter = smallBetter_order, na.treat = isna)
  } else {
    challenge <- as.challenge(data, algorithm = "algorithm", case = "case", value = "metric_value",
                              smallBetter = smallBetter_order)
  }

  if(ranking_method == "rankThenMean") {
    ranking <- challenge%>%rankThenAggregate(FUN = mean, ties.method = "min")
  } else if(ranking_method == "rankThenMedian") {
    ranking <- challenge%>%rankThenAggregate(FUN = median, ties.method = "min")
  } else if(ranking_method == "aggregateThenMean") {
    ranking <- challenge%>%aggregateThenRank(FUN = mean, na.treat = isna, ties.method = "min")
  } else if(ranking_method == "aggregateThenMedian") {
    ranking <- challenge%>%aggregateThenRank(FUN = median, na.treat = isna, ties.method = "min")
  } else if(ranking_method == "testBased") {
    ranking <- challenge%>%testThenRank(alpha = 0.05,
                                        p.adjust.method = "none",
                                        na.treat = isna, ties.method = "min")
  } else {
    warning("Please specify valid ranking scheme")
  }


  if (!is.null(report_dir)){
    # Bootstrapping analysis
    registerDoParallel(cores = 8)
    set.seed(1)
    ranking_bootstrapped <- ranking%>%bootstrap(nboot = 1000, parallel = TRUE, progress = "none")
    stopImplicitCluster()

    # Ranking report
    ranking_bootstrapped %>%
      report(title = paste(institution_name, title_name_ending, sep=" "),
             file = file.path(report_dir, paste(institution_name, file_name_ending, sep="_")),
             format = "PDF",
             latex_engine = "pdflatex",
             clean = FALSE
      )
  }

  return(ranking)
}


# Function to calculate rankings for ET, TC, WT for Dice and HD95 ------------------

#' Calculate all 6 rankings for one institutes (Dice and HD95 for ET, TC, WT)
#'
#' @param data The underlying dataset used to calculate the ranking (data.frame)
#' @param institution_name Name of the institution (used in title) (str)
#' @param ranking_method Ranking method, choose from
#' (rankThenMean, rankThenMedian, aggregateThenMean, aggregateThenMedian, testBased)
#'
#' @return A list of the 6 ranking lists

calculate_all_rankings_per_institute <- function(data, institution_name, ranking_method, report_dir = NULL) {

  ## Enhancing tumor (ET) ##
  # Compute ET ranking for the Dice metric
  print("... calculate ET Dice ranking ...")

  data_et_dice <- subset(data, metric == "Dice_ET")
  ranking_et_dice <- calculate_sub_ranking(data_et_dice, "Dice",
                                           institution_name, ranking_method,
                                           "ET Dice", "ET_Dice",
                                           report_dir)

  # Compute ET ranking for the HD95 metric
  print("... calculate ET HD95 ranking ...")

  data_et_hd95 <- subset(data, metric == "Hausdorff95_ET")
  ranking_et_hd95 <- calculate_sub_ranking(data_et_hd95, "Hausdorff95",
                                           institution_name, ranking_method,
                                           "ET HD95", "ET_HD95",
                                           report_dir)

  ## Tumor core (TC) ##
  # Compute TC ranking for the Dice metric
  print("... calculate TC Dice ranking ...")

  data_tc_dice <- subset(data, metric == "Dice_TC")
  ranking_tc_dice <- calculate_sub_ranking(data_tc_dice, "Dice",
                                           institution_name, ranking_method,
                                           "TC Dice", "TC_Dice",
                                           report_dir)

  # Compute TC ranking for the HD95 metric
  print("... calculate TC HD95 ranking ...")

  data_tc_hd95 <- subset(data, metric == "Hausdorff95_TC")
  ranking_tc_hd95 <- calculate_sub_ranking(data_tc_hd95, "Hausdorff95",
                                           institution_name, ranking_method,
                                           "TC HD95", "TC_HD95",
                                           report_dir)

  ## Whole tumor (WT) ##
  # Compute WT ranking for the Dice metric
  print("... calculate WT Dice ranking ...")
  data_wt_dice <- subset(data, metric == "Dice_WT")
  ranking_wt_dice <- calculate_sub_ranking(data_wt_dice, "Dice",
                                           institution_name, ranking_method,
                                           "WT Dice", "WT_Dice",
                                           report_dir)

  # Compute WT ranking for the HD95 metric
  print("... calculate WT HD95 ranking ...")

  data_wt_hd95 <- subset(data, metric == "Hausdorff95_WT")
  ranking_wt_hd95 <- calculate_sub_ranking(data_wt_hd95, "Hausdorff95",
                                           institution_name, ranking_method,
                                           "WT HD95", "WT_HD95",
                                           report_dir)

  # Store all rankings in a list
  rankings <- list(ranking_et_dice, ranking_et_hd95, ranking_tc_dice,
                   ranking_tc_hd95, ranking_wt_dice, ranking_wt_hd95)

  return(rankings)
}

# Function to calculate the number of significant superiorities per ranking --------

#' Overall function to calculate the number of significant superiorities per ranking
#'
#' @param rankings All sub-rankings per institute (list of ranking objects)
#' @param dataSignCounts Data frame to store significance counts
#'
#' @return Updated dataSignCount

calculate_significance_one_institute <- function(rankings, dataSignCounts) {
  print("... calculating significance counts ...")
  alpha=0.05
  p.adjust.method="holm"
  order=FALSE

  signMatrix = NULL
  for (ranking in rankings) {
    currSignMatrix = ranking$data%>%decision.challenge(na.treat=ranking$call[[1]][[1]]$na.treat,
                                                   alpha=alpha,
                                                   p.adjust.method=p.adjust.method)
    if (is.null(signMatrix)){
      signMatrix <- currSignMatrix
    }
    else {
      assertthat::are_equal(rownames(signMatrix$dummyTask), rownames(currSignMatrix$dummyTask))
      signMatrix$dummyTask <- signMatrix$dummyTask + currSignMatrix$dummyTask
    }
  }

  return(signMatrix)
}


# Load data ---------------------------------------------------------------
#' Title
#'
#' @param path Path to yaml file (str)
#'
#' @return data in data.frame format

load_data <- function(path) {

  print("... load data from institute ...")

  # Load data from yaml file and convert to data frame
  yaml_data <- yaml.load_file(path)
  # need to replace nulls from yaml, as these indicate missing values
  yaml_data <- replace_nulls_in_list(yaml_data)
  yaml_data_df <- data.frame(melt(yaml_data))

  data <- data.frame(case = yaml_data_df$L1,
                          # region = yaml_data_df$L3,  # Included in metric now
                          algorithm = yaml_data_df$L2,
                          metric = yaml_data_df$L3,
                          metric_value = yaml_data_df$value)

  return(data)
}


# couldn't find a function from the library that does this
replace_nulls_in_list <- function(x) {
  for (i in seq_along(x)) {
    value <- x[[i]]
    if (is.list(value)) {
      x[[i]] <- replace_nulls_in_list(value)
    } else {
      if (is.null(value)) {
        x[[i]] <- NA
      }
    }
  }
  x
}


# Function to calculate the mean ranks per algorithm for one institution --------

#' Overall function to compute the rankings per institute and calculate the
#' mean rank per algorithm
#'
#' @param data The underlying dataset used to calculate the ranking (data.frame)
#' @param institution_name Name of the institution (used in title) (str)
#'
#' @return Mean ranks for each algorithm (data.frame)

calculate_mean_ranks_one_institute <- function(rankings, data, institution_name, report_dir = NULL) {

  ## Bring all ranks together for each algorithm
  print("... compute mean ranks per algorithm ...")

  algorithms <- unique(data$algorithm)
  all_ranks_df <- data.frame(matrix(ncol = length(algorithms), nrow = 6))
  counter = 1

  for(alg in algorithms) {
    alg_ranks <- c()

    # Extract ranks from each of the 6 rankings for each algorithm
    for(ranking in rankings) {
      alg_rank <- ranking[[1]]$dummyTask[c(alg),c("rank")]
      alg_ranks <- rbind(alg_ranks, alg_rank)
    }

    # Store ranks for each algorithm in data frame
    all_ranks_df[[counter]] <- alg_ranks
    colnames(all_ranks_df)[counter] <- alg
    counter = counter + 1
  }

  # Compute mean rank over the 6 ranks per algorithm for this institution
  mean_rank_df <- data.frame(t(colMeans(all_ranks_df)))

  sprintf("... done with %s ...", institution_name)

  return(mean_rank_df)
}


# Main script --------------------------------------------------------------
args = commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("Please specify these arguments: data_path [, ranking_method, --make_reports].")
}
make_reports = FALSE
data_path <- args[1]
make_reports = FALSE
ranking_method <- "rankThenMean"
all_ranking_methods = list("rankThenMean", "rankThenMedian", "aggregateThenMean", "aggregateThenMedian", "testBased")

if (length(args) == 2) {
  if (args[2] == "--make_reports") {
    make_reports = TRUE
  } else {
    ranking_method <- args[2]
  }
} else if (length(args) == 3) {
  ranking_method <- args[2]
  if (!(ranking_method %in% all_ranking_methods)) {
    stop(paste("Ranking method must be one of", all_ranking_methods))
  }
  if (args[3] == "--make_reports") {
    make_reports = TRUE
  }  else {
    stop(paste("Unrecognized argument.", args[3]))
  }
}

output_dir <- "ranking_output"
if (! dir.exists(output_dir)) {
  dir.create(output_dir)
}
if (make_reports) {
  report_dir <- paste(output_dir, paste("reports",ranking_method, sep = "_"), sep = "/")
} else {
  report_dir <- NULL
}

# get list of all institution files
data_files <- list.files(data_path, pattern = '.*\\.(yaml|yml)$', full.names = TRUE)

mean_ranks_all_institutions <- NULL
all_institution_names <- NULL
all_data <- list()
dataSignMatrices <- list()

for (path in data_files) {
  # Institution i ----------------------------------------------------------
  print(path)
  institution_name <- unlist(strsplit(tail(unlist(strsplit(path, "/")), 1), "[.]"))[1]
  # print(institution_name)
  # if (institution_name == "C22_validation") {
  #   next
  #   print("skipping")
  # }
  data_fets_inst <- load_data(path)
  # data_fets_inst <- subset(data_fets_inst, algorithm != "baseline_nnunet2020")   # not ranked

  # Calculate the rankings for the ET, TC and WT
  # For each region, the ranking is computed for the Dice and Hausdorff95 metrics
  # Resulting in 6 rankings
  print("... calculate rankings ... ...")
  rankings <- calculate_all_rankings_per_institute(data_fets_inst, institution_name, ranking_method, report_dir=report_dir)

  # Compute mean rank per algorithm for each institution --------------------
  mean_rank_df <- calculate_mean_ranks_one_institute(rankings, data_fets_inst, institution_name)

  # Make sure that data frames have same ordering
  mean_rank_df %>% select(sort(names(.)))

  if (is.null(mean_ranks_all_institutions))
  {
    mean_ranks_all_institutions <- mean_rank_df
    all_institution_names <- c(institution_name)
  }
  else
  {
    mean_ranks_all_institutions <- rbind(mean_ranks_all_institutions, mean_rank_df)
    all_institution_names <- c(all_institution_names, institution_name)
  }
  all_data[[institution_name]] <- data_fets_inst

  # Calculate number of significantly superior rankings per algorithm
  dataSignMatrices[[length(dataSignMatrices) + 1]] <- calculate_significance_one_institute(rankings, dataSignCounts)
}
rownames(mean_ranks_all_institutions) <- all_institution_names

# Compute final ranking ---------------------------------------------------

final_ranks_df <- data.frame(meanRank = colMeans(mean_ranks_all_institutions))
final_ranks_df <- cbind(final_ranks_df, finalRank = rank(final_ranks_df$meanRank))
final_ranks_df <- final_ranks_df[order(final_ranks_df$finalRank),]

final_ranks_df_print <-
  hux(final_ranks_df) %>%
  add_rownames() %>%
  set_bold(row = 1, col = everywhere, value = TRUE) %>%
  set_all_borders(TRUE)

print("The final ranking is: ")
print_screen(final_ranks_df_print)
file_name_final_ranks <- paste("final_ranks", ranking_method, sep="_")
file_name_mean_ranks <- paste("per_institute_ranks", ranking_method, sep="_")
write.csv(final_ranks_df, file = paste(output_dir, paste(file_name_final_ranks, ".csv",sep=""), sep="/"))
write.csv(mean_ranks_all_institutions, file = paste(output_dir, paste(file_name_mean_ranks, ".csv",sep=""), sep="/"))

# also sum up significance matrices
total_sign_matrix <- NULL
for (s in dataSignMatrices) {
  ordered_s <- s$dummyTask[order(rownames(s$dummyTask)), order(colnames(s$dummyTask))]
  if (is_null(total_sign_matrix)){
    total_sign_matrix <- ordered_s
  } else {
    total_sign_matrix <- total_sign_matrix + ordered_s
  }
}
print("Counting how often algorithms are significantly superior to the others (each row shows the no. superiorities of that model): ")
print(total_sign_matrix)
print("Sum along rows:")
print(rowSums(total_sign_matrix))
file_name <- paste("significant_matrix", ranking_method, sep="_")
write.csv(total_sign_matrix, file = paste(output_dir, paste(file_name, ".csv",sep=""), sep="/"))
