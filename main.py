import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jax.experimental import sparse
from jax import random
import jax.nn as jnn
import jax
from jax import scipy as jsp
import jax.scipy.linalg as jla
import jax.numpy.linalg as jnla
import jax.scipy.sparse.linalg as jspla
from jax import grad, jit, vmap
import jax.numpy as jnp
from functools import partial
import optax
import jaxopt
import jax.lax.linalg as lax_linalg
import jax.lax


from jax.config import config
config.update("jax_enable_x64", True)



@partial(jit, static_argnums=[0,2])
def arnoldi_dgks(A,v,k):
    norm=jnp.linalg.norm
    dot=jnp.dot
    eta=1.0/jnp.sqrt(2.0)

    m=len(v)
    V=jnp.zeros((m,k+1))
    H=jnp.zeros((k+1,k))
    #V[:,0]=v/norm(v)
    V = V.at[:,0].set(v/norm(v))
    for j in range(0,k):
        w=A(V[:,j])
        h=V[:,0:j+1].T @ w
        f=w-V[:,0:j+1] @ h
        s = V[:,0:j+1].T @ f
        f = f - V[:,0:j+1] @ s
        h = h + s
        beta=norm(f)
        #H[j+1,j]=beta
        H = H.at[j+1,j].set(beta)
        #V[:,j+1]=f/beta
        V = V.at[:,j+1].set(f.flatten()/beta)
    return V,H



def make_banded_matrix(m,diag,bands,rng):
    subdiags=[rng.uniform(-1,1,m) for _ in bands] + [rng.uniform(0.1,1,m) + diag] + [rng.uniform(-1,1,m) for _ in bands]
    offs = [-x for x in bands] + [0] + [x for x in bands]
    return sp.diags(subdiags,offs,shape=(m,m))




seed=20987438
rng=np.random.default_rng(seed)
m=32
diag=1.0
k=11
niter=500
bandwidth=10
plot=False
#Matrix with large band
A=make_banded_matrix(m,diag,(bandwidth,),rng).toarray()
I=jnp.eye(m)

def test(params):
    t0,t1,t2=params
    T=jnp.diag(t0,k=-1)+jnp.diag(t1,k=0)+jnp.diag(t2,k=1)
    def eval(X):
        X=X.reshape((m,m))
        BX=T@X
        return BX.reshape((m*m,))
    V,H = arnoldi_dgks(eval,I.flatten(),k)
    K=V.T@V
    print(jnp.linalg.norm(K - np.eye(K.shape[0])))

def recover_precon_matrix(params):
    t0,t1,t2=params
    T=jnp.diag(t0,k=-1)+jnp.diag(t1,k=0)+jnp.diag(t2,k=1)
    def eval(X):
        X=X.reshape((m,m))
        BX=T@X
        return BX.reshape((m*m,))
    V,H = arnoldi_dgks(eval,I.flatten(),k)
    y = V @ (V.T @ A.flatten())
    return y.reshape((m,m))






@jit
def tridiag_matvec(params,x):
    m=len(x)
    t0,t1,t2=params
    y=t1*x
    y=y.at[0:m-1].add(t2*x[1:m])
    y=y.at[1:m].add(t0*x[0:m-1])
    return y


btridiag_matvec = vmap(tridiag_matvec,in_axes=(None,1),out_axes=1)


#params= rng.uniform(-1,1,m-1), rng.uniform(4,5,m),rng.uniform(-1,1,m-1)
#t0,t1,t2 = params
#X=rng.uniform(-1,1,size=(m,3))
#T=jnp.diag(t0,k=-1)+jnp.diag(t1,k=0)+jnp.diag(t2,k=1)
#out0=btridiag_matvec(params,X)
#out1=T@X
#print(f"TRIDIAG ERROR = {jnp.linalg.norm(out0-out1)}")
@jit
def loss(params):
    #t0,t1,t2=params
    #T=jnp.diag(t0,k=-1)+jnp.diag(t1,k=0)+jnp.diag(t2,k=1)
    def eval(X):
        X=X.reshape((m,m))
        BX=btridiag_matvec(params,X)
        return BX.reshape((m*m,))
    V,H = arnoldi_dgks(eval,I.flatten(),k)
    #y = V.T @ A.reshape((m*m,))
    y = V.T @ A.flatten()
    return jnp.mean( (V@y - A.flatten())**2 )



@jit
def res(params):
    t0,t1,t2=params
    T=jnp.diag(t0,k=-1)+jnp.diag(t1,k=0)+jnp.diag(t2,k=1)
    def eval(X):
        X=X.reshape((m,m))
        BX=btridiag_matvec(params,X)
        return BX.reshape((m*m,))
    V,H = arnoldi_dgks(eval,I.flatten(),k)
    y = V.T @ A.reshape((m*m,))
    return V@y - A.flatten()





params = (jnp.ones(m-1),4*jnp.ones(m),jnp.ones(m-1))
#print(test(params))
#solver = jaxopt.LBFGS(fun=loss)
solver = jaxopt.GradientDescent(fun=loss)
#solver = jaxopt.NonlinearCG(fun=loss)
#solver = jaxopt.GaussNewton(residual_fun=res)
state = solver.init_state(params)
err=loss(params)
optim=solver.l2_optimality_error(params)
print(f"loss = {err},     optimality = {optim}")
for it in range(0,niter):
    params,state = solver.update(params,state)        
    err=loss(params)
    optim=solver.l2_optimality_error(params)
    print(f"loss = {err},     optimality = {optim}")
    if plot:
        V=recover_precon_matrix(params)
        plt.close()
        plt.imshow(V,vmin=np.amin(A),vmax=np.amax(A))
        ist=f"{it}".zfill(5)
        plt.savefig(f"matrices/{ist}.svg")








res0=[]
res1=[]
b=rng.uniform(-1,1,size=m)
print("UNPRECONDITIONED")
it=0
def callback(xk):
    global it
    global res0
    it=it+1
    r=b-A@xk
    res0.append(np.linalg.norm(r))
    print(f"it={it}      res={np.linalg.norm(r)}")
spla.gmres(A,b,restart=1,callback=callback,callback_type='x')

#print("TRIDIAG PRECONDITIONED")
#it=0
#def callback(xk):
#    global it
#    it=it+1
#    r=b-A@xk
#    print(f"it={it}      res={np.linalg.norm(r)}")
#T=btridiag_matvec(params,I.reshape((m,m)))
#spla.gmres(A,b,restart=1,callback=callback,callback_type='x',M=la.inv(T))


print("POLYNOMIAL PRECONDITIONED")
it=0
def callback(xk):
    global it
    global res1
    it=it+1
    r=b-A@xk
    res1.append(np.linalg.norm(r))
    print(f"it={it}      res={np.linalg.norm(r)}")
V=recover_precon_matrix(params)
spla.gmres(A,b,restart=1,callback=callback,callback_type='x',M=la.inv(V))



plt.close()
plt.semilogy(res0)
plt.semilogy(res1)
plt.title(f"Comparing GMRES Residual Histories. bandwidth={bandwidth}")
plt.xlabel("GMRES Iteration")
plt.ylabel("Residual")
plt.legend(["Unpreconditioned","Polynomial of Tridiagonal Preconditioner"])
plt.savefig("gmres.svg")
