library(deSolve)
library(ggplot2)

seir <- function(time, state, pars) {
  with(as.list(c(state,pars)), {
    dS <- -S*beta*I/N
    dE <- S*beta*I/N-E*k
    dI <- E*k-I*(mu+gamma)
    dR <- I*gamma
    dN <- dS+dE+dI+dR
    list(c(dS,dE,dI,dR,dN))
  })
}

N <- 82927922 #总人口
E0 <- 0 #期初潜伏数
RM0 <- 16 #期初移除数
I0 <- 114 #期初感染数
S0 <- N-I0-RM0 #期初易感人数

init <- c(S = S0, E = E0, I = I0, R = RM0, N = N)
time <- seq(0, 300, 1)
pars <- c(beta = 0.55, k=1, gamma=0.43, mu=0.02)
res.seir <- as.data.frame(lsoda(y = init, times = time, func = seir, parms = pars))

windows()
ggplot(res.seir) +
  geom_line(aes(x = time, y = S, col = '2 易感')) +
  geom_line(aes(x = time, y = E, col = '3 潜伏')) +
  geom_line(aes(x = time, y = I, col = '4 感染')) +
  geom_line(aes(x = time, y = R, col = '5 移除')) +
  geom_line(aes(x = time, y = N, col = '1 人口')) +
  theme_light(base_family = 'Kai') +
  scale_colour_manual("",values=c("2 易感" = "cornflowerblue", "3 潜伏" = "orange","4 感染" = "darkred", "5 移除" = "forestgreen", "1 人口" = "black") ) +
  scale_y_continuous('')
