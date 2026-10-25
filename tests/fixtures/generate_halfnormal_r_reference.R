# Run from the repository root. Python tests need no R installation.
library(ashr)
library(jsonlite)
stopifnot(packageVersion("ashr") == package_version("2.2.67"))
stopifnot(packageVersion("jsonlite") == package_version("2.0.0"))
family <- "halfnormal"

half_normal_second_moment <- function(fit) {
  # ashr 2.2.67 comp_postsd.tnormalmix transposes data before recycling bounds.
  # Its reported SD can change when observations are permuted. Integrate the
  # fitted mixture independently, retaining reported moments for diagnosis.
  g <- fit$fitted_g
  slabs <- which(g$sd > 0)
  prior_density <- function(theta) {
    rowSums(vapply(slabs, function(j) {
      2 * g$pi[j] * dnorm(theta, 0, g$sd[j]) * (theta >= g$a[j] & theta <= g$b[j])
    }, numeric(length(theta))))
  }
  vapply(seq_along(fit$data$x), function(i) {
    x <- fit$data$x[i]
    s <- fit$data$s[i]
    moment <- function(power) {
      f <- function(theta) theta^power * prior_density(theta) * dnorm(x, theta, s)
      integrate(f, -Inf, 0, rel.tol = 1e-10, abs.tol = 1e-13)$value +
        integrate(f, 0, Inf, rel.tol = 1e-10, abs.tol = 1e-13)$value
    }
    moment(2) / (moment(0) + sum(g$pi[g$sd == 0]) * dnorm(x, 0, s))
  }, numeric(1))
}

summarize <- function(fit, scale, penalty) {
  # ashr groups all negative components before all positive components.
  k <- length(scale) - 1L
  order <- c(1L, as.vector(rbind(seq_len(k) + 1L, seq_len(k) + k + 1L)))
  reported_mean2 <- get_pm(fit)^2 + get_psd(fit)^2
  list(x = fit$data$x, s = fit$data$s, scale = scale, penalty = penalty,
       pi = fit$fitted_g$pi[order], log_lik = fit$loglik,
       mean = get_pm(fit),
       mean2 = half_normal_second_moment(fit),
       reported_mean2 = reported_mean2,
       objective = -fit$loglik - if (penalty == 1) 0 else (penalty - 1) * log(fit$fitted_g$pi[1]))
}

scale <- c(0, .25, .75, 2, 6)
set.seed(20261022)
s <- runif(250, .3, 1.2)
theta <- sample(c(0, -2, .8, 4), 250, replace = TRUE, prob = c(.5, .15, .2, .15))
x <- theta + rnorm(250) * s
fits <- lapply(c(1, 10), function(penalty) {
  # ashr adds its own point mass for half-normals. Exclude duplicate zero SDs.
  fit <- ash(x, s, mixcompdist = family,
             mixsd = scale[-1],
             prior = "nullbiased", nullweight = penalty)
  summarize(fit, scale, penalty)
})

fixed_pi <- c(.2, .05, .05, .1, .05, .05, .2, .1, .2)
k <- length(scale) - 1L
ordered_pi <- c(fixed_pi[1], fixed_pi[seq(2, 2*k, 2)], fixed_pi[seq(3, 2*k+1, 2)])
g <- tnormalmix(pi = ordered_pi, mean = rep(0, 2*k+1), sd = c(0, scale[-1], scale[-1]),
                 a = c(rep(-Inf, k+1), rep(0, k)), b = c(rep(0, k+1), rep(Inf, k)))

fixed <- ash(c(-8, -2, -.1, 0, .5, 3, 10), c(.7, .3, 1, 2, .5, 1, .8),
             g = g, fixg = TRUE, mixcompdist = family)

write_json(list(ashr_version = as.character(packageVersion("ashr")),
                jsonlite_version = as.character(packageVersion("jsonlite")),
                second_moment_reference = "Independent integration of the R fitted prior",
                seed = 20261022, fixed = summarize(fixed, scale, 1), fits = fits),
           paste0("tests/fixtures/", family, "_r_reference.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = NA)
