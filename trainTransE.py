import config
import models

con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_out_path("./benchmarks/FB15K/")
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_bern(1)
con.set_dimension(100)
con.set_margin(1)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_optimizer("SGD")
con.init()
con.set_model(models.TransE)
con.train()
