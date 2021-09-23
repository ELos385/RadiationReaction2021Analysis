import sys
sys.path.append('../../')
import matplotlib.pyplot as plt
import numpy as np


def correlationPlot_mod(df,parameters,sScale,fig=None,cmap='viridis',
                        p0 = None,pLabels=None,
                        xlims = None, ylims=None):
    """df is pandas df
    """
    if pLabels is None:
        pLabels = parameters
    if fig is None:
        fig = plt.figure(figsize=(12,8))
    N =len(parameters)
    
    if xlims is None:
        xlims = [[None,None]]*N
    if ylims is None:
        ylims = [[None,None]]*N
    ax = []
    vmax = np.max(df['fitness'].values)*1.1
    fitInd = np.argsort(df['fitness'].values)
    fitVals = df['fitness'].values[fitInd]
    n=0
    for n in range(N-1): # x-parameter index
        for m in range(n+1,N): # y-parameter index
            pInd = (n+1) + (m)*(N) 
            xInd = n
            yInd = m
            ax.append(plt.subplot(N,N,pInd))
            x = df[parameters[xInd]].values[fitInd]/sScale[xInd]
            y = df[parameters[yInd]].values[fitInd]/sScale[yInd]
            if p0 is not None:
                x = x- p0[xInd]/sScale[xInd]
                y = y- p0[yInd]/sScale[yInd]
            
            ax[-1].scatter(x,y,c=fitVals,s=6,alpha=0.5,cmap=cmap,vmax=vmax)
            ax[-1].scatter([x[-1]],[y[-1]],c=[fitVals[-1]],
                                s=6,alpha=0.5,cmap=cmap,vmax=vmax,edgecolors='r')
            ax[-1].set_xlim(xlims[xInd])
            ax[-1].set_ylim(ylims[yInd])
            if n==0:
                ax[-1].set_ylabel(pLabels[yInd])
            else:
                ax[-1].set_yticks([])
            if m==(N-1):
                ax[-1].set_xlabel(pLabels[xInd])
            else:
                ax[-1].set_xticks([])

    xInd = n
    for n in range(N): # x-parameter index
#         print(1+ n + n*N)
        xInd = n
        ax.append(plt.subplot(N,N,1+ n + n*N))
        x = df[parameters[xInd]].values/sScale[xInd]

        if p0 is not None:
            x = x - p0[xInd]/sScale[xInd]
        ax[-1].scatter(x,df['fitness'].values,s=6,alpha=0.5,cmap=cmap)
        ax[-1].set_xlim(xlims[xInd])
        ax[-1].yaxis.tick_right()
        
        if n==(N-1):
            ax[-1].set_xlabel(pLabels[xInd])
        else:
            ax[-1].set_xticks([])
        ax[-1].set_ylim((0,1.1*np.max(df['fitness'].values)))
    
    return fig, ax