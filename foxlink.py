from pyspark import SparkContext, SparkConf
from classifier import naive_bayes_classifier
import math,json


f = open('/root/classifier/config.json','r')
config = json.loads(f.read())
f.close()


conf = SparkConf().setAppName('foxlink')
sc = SparkContext(conf=conf)

referring_url_metrics = sc.textFile(config['cluster_pages_classifier']['input_cluster_pages'])

#cluster pages classifier
category_clusters = naive_bayes_classifier.keywords_naive_bayes_classifier(sc,
                                                           config['cluster_pages_classifier']['training_path_cluster_classifier'],
                                                           int(math.pow(2,int(config['cluster_pages_classifier']['number_of_features_exponent']))),
                                                           referring_url_metrics,
                                                           config['cluster_pages_classifier']['prepare_training_input_cluster_page'],
                                                           config['cluster_pages_classifier']['ouput_train_cluster_page_path_parquet'],
                                                           config['cluster_pages_classifier']['output_eval_cluster_page_path_parquet'],
                                                           config['cluster_pages_classifier']['save_cluster_page_evaluation'],
                                                           config['cluster_pages_classifier']['path_to_save_cluster_pages'])

sc.stop()