## Multi-task, test-then-rank based on Wilcoxon signed rank ranking

simulate_data = function(n=50, seed=4){
  if (!requireNamespace("permute", quietly = TRUE)) install.packages("permute")

  set.seed(seed)
  strip=runif(n,.9,1)
  c_ideal=cbind(task="c_ideal",
                rbind(
                  data.frame(alg_name="A1",value=runif(n,.9,1),case=1:n),
                  data.frame(alg_name="A2",value=runif(n,.8,.89),case=1:n),
                  data.frame(alg_name="A3",value=runif(n,.7,.79),case=1:n),
                  data.frame(alg_name="A4",value=runif(n,.6,.69),case=1:n),
                  data.frame(alg_name="A5",value=runif(n,.5,.59),case=1:n)
                ))

  set.seed(1)
  c_random=data.frame(task="c_random",
                      alg_name=factor(paste0("A",rep(1:5,each=n))),
                      value=plogis(rnorm(5*n,1.5,1)),case=rep(1:n,times=5)
  )

  strip2=seq(.8,1,length.out=5)
  a=permute::allPerms(1:5)
  c_worstcase=data.frame(task="c_worstcase",
                         alg_name=c(t(a)),
                         value=rep(strip2,nrow(a)),
                         case=rep(1:nrow(a),each=5)
  )
  c_worstcase=rbind(c_worstcase,
                    data.frame(task="c_worstcase",alg_name=1:5,value=strip2,case=max(c_worstcase$case)+1)
  )
  c_worstcase$alg_name=factor(c_worstcase$alg_name,labels=paste0("A",1:5))

  data_matrix=rbind(c_ideal, c_random, c_worstcase)
}

compute_institutional_ranking = function(df)
{
  # expected columns
  rm_cols = c("WT.dice", "WT.hd95", "TC.dice", "TC.hd95", "ET.dice", "ET.hd95")
  summed_ranks = NULL
  
  for (region_metric in rm_cols)
  {
    smallBetter = TRUE
    if (grepl("hd95", region_metric, fixed = TRUE))
    {
      smallBetter = FALSE
    }
    challenge=as.challenge(df,
                           algorithm="alg_name", case="case", value=region_metric,
                           smallBetter = smallBetter)
    ranking = challenge%>%rankThenAggregate(FUN = mean,
                                            ties.method = "min")
    
    if (is.null(summed_ranks))
    {
      # dummy task is inserted by challenge automatically
      summed_ranks = ranking$matlist$dummyTask$rank
    }
    else
    {
      summed_ranks = summed_ranks + ranking$matlist$dummyTask$rank
    }
  }
  return(rank(summed_ranks, ties.method="min"))
}

## 1\. Load package

library(challengeR)

## 2\. Load data
# a) from file
data_path = '/home/maximilian/git_repos/fets-challenge/ranking/data'
data_matrix = NULL
for (file_name in dir(data_path,
                      pattern = ".*\\.csv",
                      full.names = TRUE)
)
{
  tmp_matrix = read.csv(file_name)
  if (is.null(data_matrix))
  {
    data_matrix = tmp_matrix
  }
  else
  {
    data_matrix = rbind(data_matrix, tmp_matrix)
  }
}

# b) simulate
# data_matrix = simulate_data()


## 3 Perform ranking
summed_ranks = NULL
for (client_name in unique(data_matrix$client))
{
  df = subset(data_matrix, client==client_name)
  tmp_ranking = compute_institutional_ranking(sub_matrix)  
  if (is.null(summed_ranks))
  {
    summed_ranks = tmp_ranking
  }
  else
  {
    summed_ranks = summed_ranks + tmp_ranking
  }
}
total_ranking = rank(summed_ranks, ties.method="min")
# TODO: refactor

## 4\. Perform bootstrapping

library(doParallel)
registerDoParallel(cores=2)  
set.seed(1)
ranking_bootstrapped=ranking%>%bootstrap(nboot=100, parallel=TRUE, progress = "none")
# ranking_bootstrapped=ranking%>%bootstrap(nboot=1000, parallel=TRUE, progress = "none")
stopImplicitCluster()

## 5\. Generate the report

meanRanks=ranking%>%consensus(method = "euclidean") 
meanRanks # note that there may be ties (i.e. some algorithms have identical mean rank)

ranking_bootstrapped %>% 
  report(consensus=meanRanks,
         title="multiTaskChallengeExample",
         file = "~/git_repos/fets-challenge/ranking/MultiTask_rank-then-agg", 
         format = "PDF", # format can be "PDF", "HTML" or "Word"
         latex_engine="pdflatex",#LaTeX engine for producing PDF output. Options are "pdflatex", "lualatex", and "xelatex"
         clean=TRUE #optional. Using TRUE will clean intermediate files that are created during rendering.
  )
