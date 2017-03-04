# Text_mining_Basics
Just a simple text mining example in R using kaggle Data. I based my study on the book Machine Learning With R-Second Edition (Chapter 4)
and on the paper "Text Mining Infrastructure in R", that you can find on the docs folder.

I tried allot of models and aproaches, but I allways had problems with the data size, I only have a laptop and a not very powerfull one, but It was fun.

Also some issues isinh rWeka can arise when instaling the packages, to solve this, commnd line and write:
    - sudo R CMD javareconf
 Then do to R/Rstudio and do:
    install.packages("rJava",type='source')
    install.packages("RWeka")
    
I used Rweka to produce Brigrams, did went further that that (size problems of course), You have a alternative function to obtain brigrams.

You can find this competittion here https://www.kaggle.com/c/whats-cooking.

Have fun and Share
