
import numpy as np
import operator
from functools import reduce
import matplotlib.pyplot as plt




def main():
    # read file
    # todo Marco: use ase here
    inpFile = 'in.data'
    structures = Structure.readDataFull(inpFile)

    # get a list of all elements present in the dataset
    allElems = list(reduce(operator.or_, [s.distinctElements for s in structures]))
    # should be a list of all elements present in the dataset
    print('found elements:', allElems)

    # build matrices with energy and number of elements in structure
    es = []
    ns = []
    for s in structures:
        es.append(s.energy)
        ns.append([len([x for x in s.elements if x==el]) for el in allElems])
    # es.append(0.)
    # ns.append([1 for el in allElems])

    # if all the datapoints have the same elemental composition, then we can just use this here
    # print(np.sum(es) / np.sum(ns))
    # the method below should also work just fine thanks to the regularization!


    # we determine the optimal atomic energies by finding the minimimal squared error solution to the following equations
    # (additionally the elements are divided by the number of atoms)
    #
    # | n_{1,el_1}, n_{1,el_2}, n_(1,el_{N_el}) |    | E_{el_1}      |          | E_1 |
    # | n_{2,el_1}, n_{2,el_2}, n_(2,el_{N_el}) |    | E_{el_1}      |          | E_2 |
    #              ...                                     ...             =      ...
    # | n_{N,el_1}, n_{N,el_2}, n_(N,el_{N_el}) |    | E_{el_{N_el}} |          | E_N |
    #

    # this would be the straight forward approach.
    # unfortunately it does not allow regularization
    # x, res, rnk, s = np.linalg.lstsq(np.array(ns), np.array(es), rcond=-1)
    # print(x)

    # so we just implement it ourselves
    ns = np.array(ns)
    es = np.array(es)
    # divide everything by the number of atoms
    # since we want to minimize the error in the per atom energy
    nats = np.sum(ns, axis=1)
    ns_per_atom = ns / nats[:, None]
    es_per_atom = es / nats[:]

    # Here we add a tiny bit of regularization
    lam = 1.e-7  # regularization parameter
    # least square solution is given by this here ...
    x = np.linalg.solve(ns_per_atom.T @ ns_per_atom + np.eye(len(allElems)) * lam, ns_per_atom.T @ es_per_atom)

    print()
    print('Optimal atomic energies:')
    for i, el in enumerate(allElems):
        print('    {:<2s}'.format(elementSymbols[el]), x[i])

    print()

    newes = es - ns @ x
    print('old mean energy per atom', np.mean(es / nats))
    print('new mean energy per atom', np.mean(newes / nats))
    print('old root mean squared energy', np.sqrt(np.mean((es / nats)**2)))
    print('new root mean squared energy', np.sqrt(np.mean((newes / nats)**2)))

    # histogram of the per atom energy
    # There will be multiple peaks depending on the per atom energy
    plt.hist(es / nats, bins=100)
    plt.show()
    # now the problem should be solved
    plt.hist(newes / nats, bins=100)
    plt.show()

    # substract the atomic energies from the dataset
    for s in structures:
        elcount = np.array([len([x for x in s.elements if x==el]) for el in allElems])
        s.energy = s.energy - np.dot(elcount, x)

    # todo Marco: save the new dataset
    #Structure.saveDataFull(structures, 'out.data')


if __name__ == '__main__':
    main()
