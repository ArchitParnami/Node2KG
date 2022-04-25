import matplotlib.pyplot as plt
from GraphEmbed.scripts.datafun.CommunityGraph import CommunityGraph


def community_size(communities):
    hist = {}
    for comm in communities:
        comm_size = len(comm)
        if comm_size not in hist:
            hist[comm_size] = 1
        else:
            hist[comm_size] += 1                
    hist = sorted(hist.items())    
    return hist

datasets = ['youtube', 'dblp', 'amazon', 'lj','orkut']
labels = ['YouTube', 'DBLP', 'Amazon', 'LiveJournal','Orkut']

i=5
colors = [ 'C' + str(k) for k in range(i, i+len(datasets))]  


min_size = 0
max_size = 400
dataset_size = []

for dataset in datasets:
    graph = CommunityGraph(dataset)
    communities = graph.read_communities(min_size, max_size)
    hist = community_size(communities)
    all_sizes = []
    for comm_size, count in hist:
        l = [comm_size] * count
        all_sizes.extend(l)
    dataset_size.append(all_sizes)
    
bin_size=20
bins =  range(min_size, max_size + 1, bin_size)
bin_labels = []
for i,label in enumerate(bins,1):
    if i%2==1:
        bin_labels.append(str(label)) 
    else:
        bin_labels.append('')
    
plt.hist(dataset_size, bins=bins, histtype='bar',label=labels, color=colors)
plt.ylim(0,2000)
plt.xlabel('Community Size',fontsize=12)
plt.xticks(bins,bin_labels,fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Number of Communities',fontsize=12)
#plt.title('Frequency vs Size',fontsize=12)
plt.legend(prop={'size': 9})
plt.savefig('size_analysis.png', format='png', dpi=200,bbox_inches='tight')
plt.close()

#def plot_size_trend(dataset, communities):
#    hist = community_size(communities)
#    X = []; Y = []
#    for x, y in hist:
#        X.append(x)
#        Y.append(y)
#    plt.plot(X, Y, label=dataset)

#for dataset in datasets:
#    graph = CommunityGraph(dataset)
#    communities = graph.read_communities(min_size, max_size)
#    plot_size_trend(dataset, communities)

#plt.ylim(0,1500)
#plt.xlabel('Graph Size')
#plt.ylabel('Number of Graphs')
#plt.title('Frequency vs Size')
