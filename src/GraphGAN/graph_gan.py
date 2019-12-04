import os
import collections
import tqdm
import multiprocessing
import pickle
import numpy as np
import torch 
import config
import generator
import discriminator
from src import utils
from src.evaluation import link_prediction as lp

class GraphGAN(object):
    def __init__(self):
        print("read graph")

        # n_node(5242) 是节点的数量
        # graph 是一个字典
        self.n_node, self.graph = utils.read_edges(config.train_filename, config.test_filename)
        self.root_nodes = [i for i in range(self.n_node)]

        print("reading initial embeddings ...")

        # n_node * n_emb 的矩阵
        self.node_embed_init_d = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
                                                       n_node = self.n_node,
                                                       n_embed = config.n_emb)

        self.node_embed_init_g = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
                                                       n_node = self.n_node,
                                                       n_embed = config.n_emb)
    
        # 构建或读取 BFS-trees
        self.trees = None
        if os.path.isfile(config.cache_filename):
            print("reading BFS-trees from cache ... ")
            pickle_file = open(config.cache_filename, 'rb')
            self.trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructiong BFS-trees")
            pickle_file = open(config.cache_filename, 'wb')
            if config.multi_processing:
                self.construct_trees_with_mp(self.root_nodes)
            else:
                self.trees = self.construct_trees(self.root_nodes)
            pickle.dump(self.trees, pickle_file)
            pickle_file.close()

        print("building GAN model...")

        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()


    def construct_trees(self, nodes):
        """ use BFS algorithm to construct the BFS-trees
        
        Args:
            nodes: Graph节点中的列表
    
        Return:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list:[father, child_0, child_1, ...]
        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            
            # 把每棵树的父节点设为自己
            trees[root][root] = [root]

            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def build_generator(self):
        self.generator = generator.Generator(n_node=self.n_node, node_emd_init = self.node_embed_init_g)
    
    def build_discriminator(self):
        self.discriminator = discriminator.Discriminator(n_node = self.n_node, node_emd_init=self.node_embed_init_d)

    def train(self):
        
        self.write_embeddings_to_file()
        self.evaluation(self)
        
        print("start training ... ")
        for epoch in range(config.n_epochs):
            print("epoch %d" % epoch)
            
            # 训练判别器
            center_nodes = []
            neighbor_nodes = []
            labels = []
            for  d_epoch in range(config.n_epochs_dis):
                # 每次 dis_interval 迭代都为判别器生成新节点
                if d_epoch % config.dis_interval == 0:
                    center_nodes, neighbor_nodes, labels = self.prepare_data_for_d()
                # 开始训练
                train_size = len(center_nodes)
                start_list = list(range(0, train_size, config.batch_size_dis))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + config.batch_size_dis
                    
                    loss = torch.nn.MultiLabelSoftMarginLoss(self.discriminator.score, np.array(labels[start:end])).sum(0)
                    
                    # TODO：L2正则化
                    # node_neighbor_embedding = self.discriminator
                    # node_embedding = pass
                    # bias = passe

                    # loss = torch.nn.MultiLabelSoftMarginLoss(self.discriminator.score, np.array(labels[start:end])).sum(0) + \
                    #        config.lambda_dis * (
                    #            sum(node_neighbor_embedding ** 2) / 2 +
                    #            sum(node_embedding ** 2) / 2 +
                    #            sum(bias ** 2) / 2
                    #        )

        








    def prepare_data_for_d(self):
        """为判别器提供正采样和负采样，并记录日志"""
        center_nodes = []
        neighbor_nodes = []
        labels = []

        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                pos = self.graph[i]
                neg, _ = self.sample(i, self.trees[i], len(pos), for_d = True)
                if len(pos) != 0 and neg is not None:
                    # 正采样
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # 负采样
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0]*len(neg))
        return center_nodes, neighbor_nodes, labels

    
    def sample(self, root, tree, sample_num, for_d):
        """从 BFS-tree 中采样节点
        
        Args:
            root: int, 根节点
            tree: dict, BFS-tree
            sample_num: 需要采样的数量
            for_d : bool, 样本是用在生成器还是判别器
        
        Return:
            samples: list，采样节点的索引
            paths: list, 从根节点到采样节点的路径
        """

        all_score = self.generator.all_score
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:   # 当树只有一个节点(根)时
                    return None, None
                if for_d: # 跳过单跳节点（正采样）
                    if node_neighbor == [root]:
                        # 在当前的版本 None 被返回
                        return None, None
                    if root in node_neighbor:
                        node_neighbr.remove(root)
                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0] # 选择下一个节点
                paths[n].append(next_node)
                if next_node == previous_node: # 结束条件
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths










    def write_embeddings_to_file(self):
        """把G和D的Embedding写入文件里"""
        modes = [self.generator, self.discriminator]

        for i in range(2):
            embedding_matrix = modes[i].embedding_matrix
            index = np.array(range(self.n_node)).reshape(-1,1)
            embeddings_matrix = np.hstack([index, embeddings_matrix])
            embeddings_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(self.n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)

    @staticmethod
    def evaluation(self):
        results = []
        if config.app == "link_prediction":
            for i in range(2):
                lpe = lp.LinkPredictEval(
                    config.emb_filenames[i], config.test_filename, config.test_neg_filename, self.n_node, config.n_emb)
                result = lpe.eval_link_prediction()
                results.append(config.modes[i] + ":" + str(result) + "\n")

        with open(config.result_filename, mode="a+") as f:
            f.writelines(results)
