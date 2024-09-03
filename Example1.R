# Squaring elements of a given vector

square_for <- function(x){
  # Tests
  # is.numeric(x)
  # length(x) > 0
  
  # [ToDo] Use the for loop
  n <- length(x)
  y <- vector(length = n)
  
  for(i in 1:n){ # could also do #seq_long(x)
    y[i] <- x[i]^2
  }
  
  return(y)
}

square_sapply <- function(x){
  # [ToDo] Use the sapply function
  sapply(x, \(x) {x^2})
}

square_vec <- function(x){
  # [ToDo] Use power(^) function in vector form
  x^2
}

square_vec2 <- function(x){
  # [ToDo] Use multiplication(*) function in vector form
  x * x
}

# [ToDo] Create a vector x of size 100,000 of normal variables
x <- rnorm(100000)

# [ToDo] Verify that all functions return the same output
library(testthat)
expect_equal(square_vec(x), square_sapply(x))

# [ToDo] Use microbenchmark package to compare three functions in terms of speed
library(microbenchmark)

microbenchmark(
  square_for(x),
  square_sapply(x),
  square_vec(x),
  square_vec2(x)
)
