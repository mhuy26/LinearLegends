df <- read.csv("./data/student_lifestyle_dataset.csv")

df$Student_ID <- NULL
df$Stress_Level <- NULL

df <- na.omit(df)

hours <- c(
  "Study_Hours_Per_Day",
)

X_hours <- setdiff(hours, drop_hour)

model <- lm(
  as.formula(paste("GPA ~", paste(X_hours, collapse = " + "))),
  data = df
)

summary(model)

std_res  <- stdres(model)
stud_res <- studres(model)
rstud    <- rstudent(model)

kable(head(matrix(std_res,  ncol = 2, byrow = TRUE), 10),
      caption = "First 20 standardized residuals (shown as 10 rows x 2 cols)")

kable(head(matrix(stud_res, ncol = 2, byrow = TRUE), 10),
      caption = "First 20 studentized residuals (shown as 10 rows x 2 cols)")

kable(head(matrix(rstud,    ncol = 2, byrow = TRUE), 10),
      caption = "First 20 R-student residuals (shown as 10 rows x 2 cols)")

png("residual_bars.png", width = 1800, height = 700, res = 150)
par(mfrow = c(1, 3))

barplot(std_res,
        main = "Standardized residuals",
        ylim = c(-5, 5))
abline(h = 0, lwd = 2)
abline(h = c(-3, 3), lty = 2)

barplot(stud_res,
        main = "Studentized residuals",
        ylim = c(-5, 5))
abline(h = 0, lwd = 2)
abline(h = c(-2, 2), lty = 2)

n <- nobs(model)

alpha <- 0.05
qt_cut <- qt(1 - alpha/(2*n), df = df_resid)

barplot(rstud,
        main = paste0("R-student residuals (Bonferroni α=", alpha, ")"),
        ylim = c(-5, 5))
abline(h = 0, lwd = 2)
abline(h = c(-qt_cut, qt_cut), lty = 2)

par(mfrow = c(1, 1))
dev.off()


kable(summary(influence.measures(model))$infmat,
      caption = "Influence measures (car::influence.measures)")

dfbetasPlots(model, intercept = TRUE)

influenceIndexPlot(model, main = "Influence index plot (car)")

kable(vif(model), caption = "Variance Inflation Factors (VIF)")

png("hist_and_qq.png", width = 1600, height = 700, res = 150)
par(mfrow = c(1, 2))
hist(stud_res, breaks = 30, freq = FALSE, main = "Histogram of studentized residuals",
     xlab = "Studentized residuals")
qqPlot(model, main = "QQ Plot (car::qqPlot)")
par(mfrow = c(1, 1))
dev.off()

png("residualPlot_rstudent.png", width = 1200, height = 900, res = 150)
residualPlot(model, type = "rstudent", quadratic = FALSE)
