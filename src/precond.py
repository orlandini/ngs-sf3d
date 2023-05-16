from ngsolve import BaseMatrix


class SymmetricGS(BaseMatrix):
    def __init__(self, smoother):
        super(SymmetricGS, self).__init__()
        self.smoother = smoother

    def Mult(self, x, y):
        y[:] = 0.0
        self.smoother.Smooth(y, x)
        self.smoother.SmoothBack(y, x)

    def Height(self):
        return self.smoother.height

    def Width(self):
        return self.smoother.height


def ElPatchBlocks(mesh, fes):
    blocks = []
    freedofs = fes.FreeDofs()

    #one block per element, working
    
    for el in mesh.Elements():
        eldofs = set(d for d in fes.GetDofNrs(el) if freedofs[d])
        blocks.append(eldofs)

    # one block per edge, working
    
    # for edge in mesh.edges:
    #     edgedofs = set(d for d in fes.GetDofNrs(edge) if freedofs[d])
    #     blocks.append(edgedofs)

        
    # AFW precond. it doesnt seem to work
    
    # for v in mesh.vertices:
    #     vdofs = set()
    #     for edge in mesh[v].edges:
    #         vdofs |= set(d for d in fes.GetDofNrs(edge)
    #                      if freedofs[d])
    #     blocks.append(vdofs)
        
    return blocks


def CreatePrecond(mesh, fes, a):
    print("creating blocks")
    blocks = ElPatchBlocks(mesh, fes)
    print("created {} blocks!\nsmoothing...".format(len(blocks)))
    blockjac = a.mat.CreateBlockSmoother(blocks)
    print("\rsmoothed!Creating pre...")
    pre = SymmetricGS(blockjac)
    print("\rCreated pre!")
    return pre
