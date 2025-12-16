# ------------------------------------------------------------
# Analysis like hw7.Rmd, for student_lifestyle_dataset.csv
# Outputs plots to working directory
# ------------------------------------------------------------

# Packages used in your hw7
library(MASS)   # stdres(), studres()
library(knitr)  # kable()
library(car)    # influence measures + plots, vif(), qqPlot(), residualPlot()

# 1) Load data
df <- read.csv("./data/student_lifestyle_dataset.csv")

# Drop variables you said to ignore
df$Student_ID <- NULL
df$Stress_Level <- NULL

# Drop NAs
df <- na.omit(df)

# 2) Handle perfect collinearity: hours sum to 24
hours <- c(
  "Study_Hours_Per_Day",
)

drop_hour <- "Social_Hours_Per_Day"     # baseline activity
X_hours <- setdiff(hours, drop_hour)

# 3) Fit model
model <- lm(
  as.formula(paste("GPA ~", paste(X_hours, collapse = " + "))),
  data = df
)

summary(model)

# ------------------------------------------------------------
# Residuals tables (like your hw7)
# ------------------------------------------------------------
std_res  <- stdres(model)
stud_res <- studres(model)
rstud    <- rstudent(model)

# Show first few in tables (printing all 2000 rows is pointless)
kable(head(matrix(std_res,  ncol = 2, byrow = TRUE), 10),
      caption = "First 20 standardized residuals (shown as 10 rows x 2 cols)")

kable(head(matrix(stud_res, ncol = 2, byrow = TRUE), 10),
      caption = "First 20 studentized residuals (shown as 10 rows x 2 cols)")

kable(head(matrix(rstud,    ncol = 2, byrow = TRUE), 10),
      caption = "First 20 R-student residuals (shown as 10 rows x 2 cols)")

# ------------------------------------------------------------
# Residual bar/index plots WITH cutoff lines, saved to files
# ------------------------------------------------------------
png("residual_bars.png", width = 1800, height = 700, res = 150)
par(mfrow = c(1, 3))

# Standardized residuals: common cutoff ±3
barplot(std_res,
        main = "Standardized residuals",
        ylim = c(-5, 5))
abline(h = 0, lwd = 2)
abline(h = c(-3, 3), lty = 2)

# Studentized residuals: common cutoff ±2
barplot(stud_res,
        main = "Studentized residuals",
        ylim = c(-5, 5))
abline(h = 0, lwd = 2)
abline(h = c(-2, 2), lty = 2)

# R-student: Bonferroni-style cutoff (like your hw7, but generalized)
n <- nobs(model)
p <- length(coef(model))          # includes intercept
df_resid <- n - p                 # residual df in lm

alpha <- 0.05
# two-sided Bonferroni: alpha/(2n)
qt_cut <- qt(1 - alpha/(2*n), df = df_resid)

barplot(rstud,
        main = paste0("R-student residuals (Bonferroni α=", alpha, ")"),
        ylim = c(-5, 5))
abline(h = 0, lwd = 2)
abline(h = c(-qt_cut, qt_cut), lty = 2)

par(mfrow = c(1, 1))
dev.off()

# ------------------------------------------------------------
# Influence diagnostics (car) + index plots
# ------------------------------------------------------------

# Influence measures table
kable(summary(influence.measures(model))$infmat,
      caption = "Influence measures (car::influence.measures)")

# Show plots (no saving)
dfbetasPlots(model, intercept = TRUE)

influenceIndexPlot(model, main = "Influence index plot (car)")

# ------------------------------------------------------------
# VIF (like hw7)
# ------------------------------------------------------------
kable(vif(model), caption = "Variance Inflation Factors (VIF)")

# ------------------------------------------------------------
# Normality check: histogram + QQ plot (like hw7)
# ------------------------------------------------------------
png("hist_and_qq.png", width = 1600, height = 700, res = 150)
par(mfrow = c(1, 2))
hist(stud_res, breaks = 30, freq = FALSE, main = "Histogram of studentized residuals",
     xlab = "Studentized residuals")
qqPlot(model, main = "QQ Plot (car::qqPlot)")
par(mfrow = c(1, 1))
dev.off()

# ------------------------------------------------------------
# Residual plot (like hw7)
# ------------------------------------------------------------
png("residualPlot_rstudent.png", width = 1200, height = 900, res = 150)
residualPlot(model, type = "rstudent", quadratic = FALSE)
dev.off()

cat("Wrote files:\n",
    "- residual_bars.png\n",
    "- dfbetasPlots.png\n",
    "- influenceIndexPlot.png\n",
    "- hist_and_qq.png\n",
    "- residualPlot_rstudent.png\n")
