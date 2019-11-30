
library(bnlearn)
data("alarm")


a=read.bif("C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\alarm.bif", debug = FALSE)
b=rbn(a,500)

a1=read.bif("C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\asia.bif", debug = FALSE)
a2=read.dsc("C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\asia.dsc", debug = FALSE)
a3=read.net("C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\asia.net", debug = FALSE)
b1=rbn(a1,500)
b2=rbn(a2,500)
b3=rbn(a3,500)

write.csv(asia,"C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\asia.csv", row.names = FALSE)
a1=read.bif("C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\cancer.bif", debug = FALSE)
b1=rbn(a1,100000)
write.csv(b1,"C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\cancer.csv", row.names = FALSE)

alarm1=read.bif("C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\alarm.bif", debug = FALSE)
dados_alarm=rbn(alarm1,100000)
write.csv(dados_alarm,"C:\\Users\\Itallo\\Documents\\RB_R_GERAR\\alarm.csv", row.names = FALSE)


load(path.expand("~/Documents/RB_R_GERAR/alarm.bif"))
write.csv(asia,'asia.csv')



