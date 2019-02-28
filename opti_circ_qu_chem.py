import numpy as np
from qiskit.chemistry import bksf
from qiskit.chemistry.fermionic_operator import FermionicOperator as FO
import pdb
from collections import OrderedDict
from pyscf import gto, scf, ao2mo
from pyscf.lib import param
from scipy import linalg as scila
from pyscf.lib import logger as pylogger
# from qiskit.chemistry import AquaChemistryError
from qiskit.chemistry import QMolecule
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager, transpile_dag, transpile
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua.algorithms.classical import ExactEigensolver
from qiskit.chemistry import FermionicOperator
from qiskit.quantum_info import Pauli
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import CXCancellation, Optimize1qGates
from qiskit import BasicAer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import networkx as nx
import int_func

#initialize one_body matrix and the two_body matrix

np.set_printoptions(linewidth=230,suppress=True)


########## Random Test Case ##############
two_body = np.zeros([4,4,4,4])
one_body=np.zeros([4,4])
one_body[3,2]=.5
one_body[2,3]=0.5
one_body[0,3]=0.4
one_body[3,0]=0.4
one_body[3,1]=0.3
one_body[1,3]=0.3
one_body[1,2]=0.21
one_body[2,1]=0.21
two_body=np.einsum('ijkl->iljk',two_body)
########### H2O ######################
mol = gto.Mole()
mol.atom = [['O',(0.0, 0.0,0.0)],['H',(1, 0, 0)], ['H',(-1.0,0.0,0.0)]]
mol.basis = 'sto-3g'
_q_=int_func.qmol_func(mol, atomic=True)
one_body=_q_.one_body_integrals
obs=np.size(one_body,1)
two_body=_q_.two_body_integrals

########### NH3 ######################
mol = gto.Mole()
mol.atom = [['N', (0.0000,  0.0000, 0.0000)],   
				['H', (0.0000,	-0.9377,	-0.3816)],  
				['H', (0.8121,	0.4689	,-0.3816)],  
				['H', (-0.8121,	0.4689	,-0.3816)]]
mol.basis = 'sto-3g'
_q_=int_func.qmol_func(mol, atomic=True)
one_body=_q_.one_body_integrals
obs=np.size(one_body,1)
two_body=_q_.two_body_integrals

########### CH4 ######################
mol = gto.Mole()
mol.atom=[['C',  (2.5369,    0.0000,    0.0000)],    
	         ['H',  (3.0739,    0.3100,    0.0000)],  
	         ['H',  (2.0000,   -0.3100,    0.0000)],  
	         ['H',  (2.2269,    0.5369,    0.0000)],    
	         ['H',  (2.8469,   -0.5369,    0.0000)]]   
mol.basis = 'sto-3g'
_q_=int_func.qmol_func(mol, atomic=True)
one_body=_q_.one_body_integrals
obs=np.size(one_body,1)
two_body=_q_.two_body_integrals



# def r_mat(one_body, two_body)
fer_op1=FO(h1=one_body,h2=two_body)
# First step of the algorithm is to get the graph.
edge_list=bksf.bravyi_kitaev_fast_edge_list(fer_op1)

two_body=np.zeros([obs,obs,obs,obs])
fer_op1=FO(h1=one_body,h2=np.einsum('ijkl->iljk',two_body))
#initialize graph and calculate the degree
G=nx.Graph()
G.add_edges_from(edge_list.transpose())
degree=np.zeros(np.size(one_body,1))
#print([e for e in G.edges])
for i in range(obs):
    degree[i]=G.degree(i)
# sort_deg=np.argsort(degree)

#sort the edges with degree in descending order
sort_deg=(-degree).argsort()


#calculate r matrix to transform the Hamiltonian for renumbered fermionic modes.
rmat=np.zeros([np.size(one_body,1),np.size(one_body,1)])
for i in range(np.size(one_body,1)):
    rmat[i,np.where(sort_deg==i)]=1


#initialize fermionic operator with the renumbered Hamiltonian
fer_op1.transform(rmat)
jw_qo=fer_op1.mapping('jordan_wigner')

# Checking to make sure everything went correctly.
edge_list=bksf.bravyi_kitaev_fast_edge_list(fer_op1)
G=nx.Graph()
G.add_edges_from(edge_list.transpose())
# degree=np.zeros(4)
# print(edge_list)
# # jw_qo=Operator(paulis=[[.4,Pauli.from_label('ZZZZ')],[.4,Pauli.from_label('IZZZ')]])


# Separating the x terms and the y terms in the Hamiltonian.
X_paulis=[]
Y_paulis=[]
d=0
for i in jw_qo._paulis:
    if d%2==0:
        X_paulis.append(i)
    else:
        Y_paulis.append(i)
    d=d+1

for i in X_paulis:
    Y_paulis.append(i)

# initialize the operator with reordered terms
jw_x_qo=Operator(Y_paulis)
# print(jw_x_qo.print_operators())


# Generating the circuit from the Operator class
q = QuantumRegister(np.size(one_body,1), name='q')
q_circ=jw_x_qo.evolve(None, 1, 'circuit', 1,q)


# Getting rid of extra gates
simulator = BasicAer.get_backend('qasm_simulator')
pass_manager = PassManager()
# pass_manager.append(CXCancellation())
pass_manager.append(Optimize1qGates())
# q_circ.draw()
# print(q_circ.count_ops())
new_circ = transpile(q_circ, simulator, pass_manager=pass_manager)
pass_manager=PassManager()
pass_manager.append(CXCancellation())
for i in range(np.size(one_body,1)):
    new_circ = transpile(new_circ, simulator, pass_manager=pass_manager)
#    new_circ = transpile(new_circ, simulator, pass_manager=pass_manager)

print("Gate count using the optimization based on the structure of the fermionic problem")
print(new_circ.count_ops())




# q = QuantumRegister(4, name='q')
# q_circ=jw_qo.evolve(None, 1, 'circuit', 1,q)




