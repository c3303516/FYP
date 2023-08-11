def buildHamiltonian(p,Mq,dMdq,place):
    temp = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq)))
    dMinvdq = -linalg.solve(Mq, temp)  

    pMp = jnp.transpose(p)@dMinvdq@p
    placeArray = jnp.zeros((7,1))
    placeArray = placeArray.at[[place],0].set(1.)
    # print('PlaceArray', placeArray)
    # pdot = jnp.transpose(p)@linalg.solve(Mq,placeArray) + (jnp.transpose(placeArray)@linalg.solve(Mq,p))
    return dMinvdq, pMp