import numpy as np
from sklearn.svm import LinearSVC


def split(x, y, ratio):
    perm = np.random.permutation(x.shape[0])
    num = int(x.shape[0] * ratio)
    return x[perm][:num], y[perm][:num], x[perm][num:], y[perm][num:]


class BatchGenerator:
    def __init__(self, x, y, xt, batch_size):
        print(np.sum(y, 0))
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.xt = np.copy(xt)
        self.batch_size = batch_size

    def next_batch(self):
        mask = np.random.randint(0, self.x.shape[0], size=(self.batch_size))
        x = self.x[mask]
        y = self.y[mask]

        mask = np.random.randint(0, self.xt.shape[0], size=(self.batch_size))
        xt = self.xt[mask]
        return x, y, xt

    def generate_validate(self, model):
        lt = np.argmax(model.predict(self.xt), 1)
        yt = np.zeros((self.xt.shape[0], self.y.shape[1]))
        for i in range(self.xt.shape[0]):
            yt[i, lt[i]] = 1
        print(np.sum(yt, 0))

        train_tar_x, _, test_tar_x, test_tar_y = split(self.x, self.y, 0.9)
        batches = BatchGenerator(self.xt, yt, train_tar_x, self.batch_size)
        # clf = LinearSVC()
        # m = clf.fit(self.xt, np.argmax(yt, 1))
        # train_accuracy = clf.score(self.xt, np.argmax(yt, 1))
        # test_accuracy = clf.score(test_tar_x, np.argmax(test_tar_y, 1))
        # print(train_accuracy, test_accuracy)
        return batches, test_tar_x, np.argmax(test_tar_y, 1)


class Batches3:
    def __init__(self, x, y, xt, batch_size):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.xt = np.copy(xt)
        self.batch_size = batch_size

    def next_batch(self):
        perm = np.random.permutation(self.x.shape[0])
        self.x = self.x[perm]
        self.y = self.y[perm]
        np.random.shuffle(self.xt)
        return self.x[:self.batch_size], self.y[:self.batch_size], self.xt[:self.batch_size]


class Batches4:
    def __init__(self, x, y, xt, batch_size):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.xt = np.copy(xt)
        mask = np.random.randint(0, self.xt.shape[0], size=(self.x.shape[0]))
        self.xt = self.xt[mask]
        self.batch_size = batch_size
        self.cnt = 0

    def next_batch(self):
        if self.cnt + self.batch_size < self.x.shape[0]:
            perm = np.random.permutation(self.x.shape[0])
            self.x = self.x[perm]
            self.y = self.y[perm]
            np.random.shuffle(self.xt)
            self.cnt = 0

        x = self.x[self.cnt:self.cnt+self.batch_size]
        y = self.y[self.cnt:self.cnt+self.batch_size]
        xt = self.xt[self.cnt:self.cnt+self.batch_size]
        self.cnt += self.batch_size
        return x, y, xt


class Batches:
    def __init__(self, x, y, xt, batch_size):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.xt = np.copy(xt)
        self.batch_size = batch_size
        
    def next_batch(self):
        """
        get next batch
        """
        if self.y.shape[0]>self.batch_size:
            return self.next_batch_smaller(self.x, self.y, self.batch_size)
        else:
            return self.next_batch_bigger()
            
    def next_batch_smaller(self, x, y, batch_size):
        """
        calculate random batch if batchsize of target is smaller than source
        """
        x_batch = np.array([])
        y_batch = np.array([])
        n_min = int(np.min(self.y.sum(0)))
        n_rest = int(batch_size - n_min*y.shape[1])
        if n_rest<0:
            n_min = int(batch_size /y.shape[1])
            n_rest = batch_size %y.shape[1]
        ind_chos = np.array([])
        is_first = True
        # fill with n_min samples per class
        for cl in range(y.shape[1]):
            ind_cl = np.arange(y.shape[0])[y[:,cl]!=0]
            ind_cl_choose = np.random.permutation(np.arange(ind_cl.shape[0]))[:n_min]
            if is_first:
                x_batch = x[ind_cl[ind_cl_choose]]
                y_batch = y[ind_cl[ind_cl_choose]]
                is_first = False
            else:
                x_batch = np.concatenate((x_batch,x[ind_cl[ind_cl_choose]]),axis=0)
                y_batch = np.concatenate((y_batch,y[ind_cl[ind_cl_choose]]),axis=0)
            ind_chos = np.concatenate((ind_chos,ind_cl[ind_cl_choose]))
        # fill with n_rest random samples
        mask = np.ones(x.shape[0],dtype=bool)
        mask[ind_chos.astype(int)] = False
        x_rem = x[mask]
        y_rem = y[mask]
        ind_choose = np.random.permutation(np.arange(x_rem.shape[0]))[:n_rest]
        x_batch = np.concatenate((x_batch,x_rem[ind_choose]),axis=0)
        y_batch = np.concatenate((y_batch,y_rem[ind_choose]),axis=0)
        return x_batch, y_batch, self.xt
        
    def next_batch_bigger(self):
        """
        calculate random batch if batchsize of target is greater than source
        """
        n_remaining = self.batch_size
        is_first = True
        while n_remaining >= self.x.shape[0]:
            if is_first:
                x_batch = self.x
                y_batch = self.y
                is_first = False
            else:
                x_batch = np.concatenate((x_batch,self.x),axis=0)
                y_batch = np.concatenate((y_batch,self.y),axis=0)
            n_remaining -= self.x.shape[0]
        x_add, y_add = self.next_batch_smaller(self.x, self.y, n_remaining)
        x_batch = np.concatenate((x_batch,x_add),axis=0)
        y_batch = np.concatenate((y_batch,y_add),axis=0)
        return x_batch, y_batch, self.xt


