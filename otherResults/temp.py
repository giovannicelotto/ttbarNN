import numpy as np
import matplotlib.pyplot as plt

#learningRate =  0.0001
mean = -4
sigma = 0.666
#mu = np.log(mean**2/np.sqrt(mean**2 + sigma**2))
#sigma_norm = np.sqrt(np.log(1 + (sigma**2 / mean**2)))
lrlist = []
for i in range(10000):
    lR = np.random.normal(loc = mean, scale = sigma)
    lrlist.append(10**(lR))
lrlist = np.array(lrlist)
fig, ax =plt.subplots(1, 1)
ax.hist(lrlist, bins = np.logspace(-7, 4, num=100), )
ax.set_xscale('log')


#m1 = (lrlist>10**-2.6666) & (lrlist<10**-1.333)
#m2 = (lrlist>10**-6) & (lrlist<10**-2)
#m3 = (lrlist>10**-4) & (lrlist<10**0)


#print(len(lrlist[m1])/len(lrlist))
#print(len(lrlist[m2])/len(lrlist))
#print(len(lrlist[m3])/len(lrlist))




fig.savefig("/nfs/dust/cms/user/celottog/mttNN/otherResults/pic.png")
