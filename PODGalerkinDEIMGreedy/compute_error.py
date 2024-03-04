import numpy as np

time=np.load('time.npy')
import pyvista

u_train_true=np.zeros((50,1472))
u_test_true=np.zeros((50,1472))
u_train_pred=np.zeros((50,1472))
u_test_pred=np.zeros((50,1472))

for i in range(50):
    reader = pyvista.get_reader('../Train/NonLinElliptic/snapshots/truth_{}.xdmf'.format(i))
    u_train_true[i,:]=reader.read().point_data[reader.read().point_data.keys()[0]]
for i in range(50):
    reader = pyvista.get_reader('../Test/NonLinElliptic/snapshots/truth_{}.xdmf'.format(i))
    u_test_true[i,:]=reader.read().point_data[reader.read().point_data.keys()[0]]

for i in range(50):
    reader = pyvista.get_reader('./NonLinElliptic/online_solution_train_{}.xdmf'.format(i))
    u_train_pred[i,:]=reader.read().point_data[reader.read().point_data.keys()[0]]

for i in range(50):
    reader = pyvista.get_reader('./NonLinElliptic/online_solution_test_{}.xdmf'.format(i))
    u_test_pred[i,:]=reader.read().point_data[reader.read().point_data.keys()[0]]

print("{:.2e}".format((time)))
print("{:.2e}".format(np.linalg.norm(u_train_true-u_train_pred)/np.linalg.norm(u_train_true)))
print("{:.2e}".format(np.linalg.norm(u_test_true-u_test_pred)/np.linalg.norm(u_test_true)))