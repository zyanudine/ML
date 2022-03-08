install.packages("binaryLogic",repos="http://cran.us.r-project.org")
library(binaryLogic)
as.binary(5,n=20)
negate(as.binary(5,n=20))

# 1/0 inverse operation
> a
[1] 1 0 1 0
> b
[1] 1 1 0 0
> abs(a-1)
[1] 0 1 0 1
> abs(b-1)
[1] 0 0 1 1
> a*b+abs(a-1)*abs(b-1)
[1] 1 0 0 1
> a=c(1,0,1,0,1,1,1,0,0,0)
> b=c(1,1,0,0,0,0,0,1,1,1)
> a*b+abs(a-1)*abs(b-1)
 [1] 1 0 0 1 0 0 0 0 0 0

