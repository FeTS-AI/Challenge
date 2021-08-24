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
#' @param title_name_ending Ending of the title string (str)
#' @param file_name_ending Ending of the file name string (str)
#'
#' @return A ranking list

calculate_sub_ranking <- function(data, metric_variant, institution_name, 
                                  title_name_ending, file_name_ending, report_dir = NULL) {
  
  smallBetter_order <- FALSE
  if(metric_variant == "Hausdorff95") {
    smallBetter_order <- TRUE
  }
  
  challenge <- as.challenge(data, algorithm = "algorithm", case = "case", value = "metric_value", 
                            smallBetter = smallBetter_order)
  
  ranking <- challenge%>%rankThenAggregate(FUN = mean, ties.method = "min")
  
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
#'
#' @return A list of the 6 ranking lists

calculate_all_rankings_per_institute <- function(data, institution_name, report_dir = NULL) {
  
  ## Enhancing tumor (ET) ##
  data_et <- subset(data, region == "ET")
  
  # Compute ET ranking for the Dice metric
  print("... calculate ET Dice ranking ...")
  
  data_et_dice <- subset(data_et, metric == "Dice")
  ranking_et_dice <- calculate_sub_ranking(data_et_dice, "Dice", 
                                           institution_name, "ET Dice", "ET_Dice",
                                           report_dir)
  
  # Compute ET ranking for the HD95 metric
  print("... calculate ET HD95 ranking ...")
  
  data_et_hd95 <- subset(data_et, metric == "Hausdorff95")
  ranking_et_hd95 <- calculate_sub_ranking(data_et_hd95, "Hausdorff95", 
                                           institution_name, "ET HD95", "ET_HD95",
                                           report_dir)
  
  ## Tumor core (TC) ##
  data_tc <- subset(data, region == "TC")
  
  # Compute TC ranking for the Dice metric
  print("... calculate TC Dice ranking ...")
  
  data_tc_dice <- subset(data_tc, metric == "Dice")
  ranking_tc_dice <- calculate_sub_ranking(data_tc_dice, "Dice", 
                                           institution_name, "TC Dice", "TC_Dice",
                                           report_dir)
  
  # Compute TC ranking for the HD95 metric
  print("... calculate TC HD95 ranking ...")
  
  data_tc_hd95 <- subset(data_tc, metric == "Hausdorff95")
  ranking_tc_hd95 <- calculate_sub_ranking(data_tc_hd95, "Hausdorff95", 
                                           institution_name, "TC HD95", "TC_HD95",
                                           report_dir)
  
  ## Whole tumor (WT) ##
  data_wt <- subset(data, region == "WT")
  
  # Compute WT ranking for the Dice metric
  print("... calculate WT Dice ranking ...")
  data_wt_dice <- subset(data_wt, metric == "Dice")
  ranking_wt_dice <- calculate_sub_ranking(data_wt_dice, "Dice", 
                                           institution_name, "WT Dice", "WT_Dice",
                                           report_dir)
  
  # Compute WT ranking for the HD95 metric
  print("... calculate WT HD95 ranking ...")
  
  data_wt_hd95 <- subset(data_wt, metric == "Hausdorff95")
  ranking_wt_hd95 <- calculate_sub_ranking(data_wt_hd95, "Hausdorff95", 
                                           institution_name, "WT HD95", "WT_HD95",
                                           report_dir)
  
  # Store all rankings in a list
  rankings <- vector(mode = "list", length = 6)
  
  rankings[1] <- ranking_et_dice
  rankings[2] <- ranking_et_hd95
  rankings[3] <- ranking_tc_dice
  rankings[4] <- ranking_tc_hd95
  rankings[5] <- ranking_wt_dice
  rankings[6] <- ranking_wt_hd95
  
  return(rankings)
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
  yaml_data_df <- data.frame(melt(yaml_data))
  
  data <- data.frame(case = yaml_data_df$L1, 
                          region = yaml_data_df$L3,
                          algorithm = yaml_data_df$L2,
                          metric = yaml_data_df$L4,
                          metric_value = yaml_data_df$value)
  
  return(data)
}


# Function to calculate the mean ranks per algorithm for one institution --------

#' Overall function to compute the rankings per institute and calculate the 
#' mean rank per algorithm
#'
#' @param data The underlying dataset used to calculate the ranking (data.frame)
#' @param institution_name Name of the institution (used in title) (str)
#'
#' @return Mean ranks for each algorithm (data.frame)

calculate_mean_ranks_one_institute <- function(data, institution_name, report_dir = NULL) {
  
  # Calculate the rankings for the ET, TC and WT
  # For each region, the ranking is computed for the Dice and Hausdorff95 metrics
  # Resulting in 6 rankings
  print("... calculate rankings ... ...")
  
  rankings <- calculate_all_rankings_per_institute(data, institution_name, report_dir)
  
  ## Bring all ranks together for each algorithm
  print("... compute mean ranks per algorithm ...")
  
  algorithms <- unique(data$algorithm)
  all_ranks_df <- data.frame(matrix(ncol = length(algorithms), nrow = 6))
  counter = 1
  
  for(alg in algorithms) {
    alg_ranks <- c()
    
    # Extract ranks from each of the 6 rankings for each algorithm
    for(ranking in rankings) {
      alg_rank <- ranking$dummyTask[c(alg),c("rank")]
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


# Function to generate boxplots -------------------------------------------

#' Generates a dot- and boxplot for the raw metric values of one institute, 
#' grouped by the regions ET, TC and WT for one metric
#'
#' @param data Raw data used to generate plots (data.frame)
#' @param metric_variant Either "Dice" or "Hausdorff95" (str)  
#' @param institution_name Name of the institution (used in title) (str)
#'
#' @return Ggplot

generate_dot_boxplots_per_institute <- function(data, metric_variant, institution_name) {

  p <- ggplot(data, aes(x=algorithm, y=metric_value, color=region)) +
    geom_boxplot(lwd=0.8,outlier.shape=NA) + 
    geom_point(position=position_jitterdodge(), alpha = 0.8, size=1.3) +
    xlab("Algorithm") + ylab(metric_variant) + labs(color = "Region") +
    theme_light() + ggtitle(institution_name) + 
    theme(axis.text.x = element_text(size = 12, angle = 90, vjust = 0.5, hjust=1),
          axis.text.y = element_text(size = 13),
          legend.text = element_text(size = 13),
          legend.title = element_text(size= 16),
          axis.title = element_text(size= 16),
          strip.text = element_text(size = 16),
          title = element_text(size=16),
          legend.position="bottom"
    )
  
  return(p)
}

# Main script --------------------------------------------------------------
args = commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("Please specify these arguments: data_path [, report_save_dir].")
} else if (length(args) == 1) {
  report_dir = NULL
} else if (length(args) == 2) {
  report_dir = args[2]
}
data_path <- args[1]

# get list of all institution files
data_files <- list.files(data_path, pattern = '.*\\.(yaml|yml)$', full.names = TRUE)

mean_ranks_all_institutions <- NULL
all_institution_names <- NULL
all_data <- list()
for (path in data_files) {
  # Institution i ----------------------------------------------------------
  print(path)
  institution_name <- unlist(strsplit(tail(unlist(strsplit(path, "/")), 1), "[.]"))[1]
  data_fets_inst <- load_data(path)
  
  # Compute mean rank per algorithm for each institution --------------------
  mean_rank_df <- calculate_mean_ranks_one_institute(data_fets_inst, institution_name, report_dir)
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
