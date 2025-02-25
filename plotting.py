import os 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

# plotting 
def plotting(plt, data, colori, pltLabel):
    plt.hold(True)
    
    for i in range (np.shape(data)[0]):
        plt.subplot(4,5,5*i+1)
        plt.plot(np.transpose(data[i])[0,:], np.transpose(data[i])[1,:],colori,label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('IL')	
        plt.grid(True)


        plt.subplot(4,5,5*i+2)
        plt.plot(np.transpose(data[i])[0,:], np.transpose(data[i])[2,:],colori, label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('OO')
        plt.grid(True)
        
        plt.subplot(4,5,5*i+3)
        plt.plot(np.transpose(data[i])[0,:], np.transpose(data[i])[3,:],colori, label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('a')
        plt.grid(True)
        
        plt.subplot(4,5,5*i+4)
        plt.plot(np.transpose(data[i])[0,:], np.transpose(data[i])[5,:],colori,label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('OUTL')
        plt.grid(True)

        plt.subplot(4,5,5*i+5)
        plt.plot(np.transpose(data[i])[0,:], -1*np.transpose(data[i])[4,:],colori,label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('r')
        plt.grid(True)

    return plt
    
def savePlot(players, curGame, Rsltdnn, RsltFrmu, RsltOptm, config, m):
    node1 = config.node1
    node2 = config.node2
    node3 = config.node3
    #add title to plot
    if config.if_titled_figure:
        if config.NoHiLayer==2:
            plt.suptitle("Game No="+str(curGame)+";" + str(config.agentTypes.count("srdqn"))+ " SRDQN Agents; SRDQN nodes="+str(node1)+
            "-"+str(node2)+ "; sum SRDQN=" + str(round(sum(Rsltdnn),2)) + "; sum Strm=" 
            + str(round(sum(RsltFrmu),2))  +"; sum BS=" +  str(round(sum(RsltOptm),2))+ "\n"+
            "Ag SRDQN="+str([round(Rsltdnn[i],2) for i in range(config.NoAgent)])+
            "; Ag Strm="+str([round(RsltFrmu[i],2) for i in range(config.NoAgent)])+
            "; Ag BS="+str([round(RsltOptm[i],2) for i in range(config.NoAgent)]), fontsize=12)
        elif config.NoHiLayer==3:
            plt.suptitle("Game No="+str(curGame)+";" + str(config.agentTypes.count("srdqn"))+ " SRDQN Agents; SRDQN nodes="+str(node1)+
            "-"+str(node2)+"-"+str(node3)+ "; sum SRDQN=" +  str(round(sum(Rsltdnn),2))  + 
            "; sum Strm=" +  str(round(sum(RsltFrmu),2))  +"; sum BS=" +  str(round(sum(RsltOptm),2))+"\n"+
            "Ag SRDQN="+str([round(Rsltdnn[i],2) for i in range(config.NoAgent)])+
            "; Ag Strm="+str([round(RsltFrmu[i],2) for i in range(config.NoAgent)])+
            "; Ag BS="+str([round(RsltOptm[i],2) for i in range(config.NoAgent)]), fontsize=12)
                    
            
    #insert legend to the figure
    legend = plt.legend(bbox_to_anchor=(-1.4, -.165, 1., -.102), shadow=True, ncol=4)

    # configures spaces between subplots
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=.5, hspace=.5)
    # save the figure
    plt.savefig(os.path.join(config.model_dir,'saved_figures/') + str(curGame)+ '-' + str(m)+'.pdf', format='pdf')
    print("figure"+str(curGame)+".pdf saved in folder \"saved_figures\"")
    plt.close(curGame)


def plotBaseStock(data, colori, pltLabel, curGame, config, m):
    plt.figure(104, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')	
    plt.plot(range(len(data)), data, colori, label=pltLabel)
    plt.xlabel('Time')
    plt.ylabel('Order-up-to level')
    plt.grid(True)
    plt.savefig(os.path.join(config.model_dir,'saved_figures/') + "dnnBaseStock" + str(curGame)+ '-' + str(m)+'.pdf', format='pdf')
    print("base stock figure"+str(curGame)+ '-' + str(m)+".pdf saved in folder \"saved_figures\"")
    plt.close(104)
