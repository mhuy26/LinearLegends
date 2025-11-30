df <- read.csv("student_lifestyle_dataset.csv")
df$stress_numeric <- as.numeric(factor(df$Stress_Level, levels = c("Low", "Moderate", "High")))

fit <- lm(GPA ~ Study_Hours_Per_Day, data = df)

anova(fit)
summary(fit)


set.seed(10)

for(k in 1:5)
{
  i <- sample(1:nrow(df), 1)
  row_i <- df[i, ]
  
  newdata <- data.frame(
    Study_Hours_Per_Day = row_i$Study_Hours_Per_Day
  )
  
  interval <- predict(fit, newdata, interval = "prediction", level = 0.95)
  in_interval <- row_i$GPA >= interval[,"lwr"] & row_i$GPA <= interval[,"upr"]
  
  cat("Row:", i, "\n")
  cat("Fit:", interval[,"fit"], "\n")
  cat("95% PI:", interval[,"lwr"], "to", interval[,"upr"], "\n")
  cat("Actual GPA:", row_i$GPA, "\n")
  cat("Is actual GPA in interval? ", in_interval, "\n")
  cat("Residual value: ", row_i$GPA - interval[,"fit"], "\n\n")
}


student1_data <- data.frame(
  Study_Hours_Per_Day = 7.5
)

interval <- predict(fit, student1_data, interval = "prediction", level = 0.95)

cat("Fit:", interval[,"fit"], "\n")
cat("95% PI:", interval[,"lwr"], "to", interval[,"upr"], "\n")


#Model analysis:
summary(fit)
anova(fit)