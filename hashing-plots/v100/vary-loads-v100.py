# coding: utf-8

# In[19]:

import sys
import pandas as pd
import numpy
import itertools
import toyplot
import toyplot.pdf
#get_ipython().run_line_magic('matplotlib', 'inline')


filename = sys.argv[1]
col_0_name = "numPairs"
y_label_0 = "Insertion Throughput (billions pairs/sec)"
x_label_0 = "Load Factor"
title_0 = "K40 Insertions, 300M Pairs"
y_label_1 = "Query Throughput (billions pairs/sec)"
x_label_1 = "Load Factor"
title_1 = "K40 Querying, 300M Pairs"


df = pd.read_csv(filename, header=None, names=['numPairs', 'hfInsert', 'hfQuery', 'cuInsert', 'cuQuery', 'thSort', 'thSearch', 'hfInsert2', 'hfQuery2', 'cuInsert2', 'cuQuery2', 'thSort2', 'thSearch2'])


df['hfInsert'] = 900000000/df.hfInsert
df['cuInsert'] = 900000000/df.cuInsert
df['thSort'] = 900000000/df.thSort
df['hfInsert2'] = 300000000/df.hfInsert2
df['cuInsert2'] = 300000000/df.cuInsert2
df['thSort2'] = 300000000/df.thSort2

# In[29]:
df['hfQuery'] = 900000000/df.hfQuery
df['cuQuery'] = 900000000/df.cuQuery
df['thSearch'] = 900000000/df.thSearch
df['hfQuery2'] = 300000000/df.hfQuery2
df['cuQuery2'] = 300000000/df.cuQuery2
df['thSearch2'] = 300000000/df.thSearch2

colors = ["green", "steelblue", "darkred"]
markers = ["o", "--", "^"]
exts = ["", "2"]

def GetStyle(toplot) :
  style = {}
  keys = algos.keys()
  for key in keys:
    if key in toplot :
      index = algos.get(key)
      style["stroke"] = colors[index]
    #if "2" in toplot :
      #style["stroke-dasharray"] = "2, 2"
  return style

#axes1 = canvas.cartesian(grid=(1, 2, 0))
#axes1.plot(y)
#axes2 = canvas.cartesian(grid=(1, 2, 1))
#axes2.plot(1 - y);

canvas = toyplot.Canvas('3.25in', '6in')


#Plot 3: V100 Insertions
algos = {"hfInsert":0, "cuInsert":1, "thSort":2}
legends = {"hfInsert":"HashFight", \
           "cuInsert":"CUDPP", \
           "thSort":"Thrust-Sort"}

axes3 = canvas.cartesian(label = "V100 Insertions, 900M Pairs",
                        xlabel = 'Load Factor',
                        ylabel = 'Insert Throughput (1M pairs/sec)',
                        grid = (2,1,0),
                        margin = (40,50,50))
axes3.label.style = {"font-size":"12px"}
axes3.x.label.style = {"font-size":"8px"}
axes3.y.label.style = {"font-size":"8px"}
axes3.x.ticks.show = True
axes3.y.ticks.show = True
axes3.y.ticks.labels.angle = -45
axes3.x.ticks.far = "4px"
axes3.y.ticks.far = "4px"
axes3.y.label.location = "above"
axes3.y.label.offset = "30px"
axes3.x.ticks.locator = toyplot.locator.Explicit([1, 1.2, 1.4, 1.6, 1.8, 2.0])
axes3.y.ticks.locator = toyplot.locator.Explicit([0, 500, 1000, 1500, 2000, 2500, 3000])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    x = df.numPairs.values
    y = numpy.array(data)
    plotstyle = GetStyle(toplot)
    label = legends.get(toplot)
    thisMarker = markers[algos.get(algo)]
    thisStroke = plotstyle.get("stroke")
    axes3.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes3._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "middle"
    #if "cu" in algo:
      #vertalign = "first-baseline"
    axes3.text(x[-1], y[-1], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"7.5px", \
                                          "-toyplot-vertical-align":vertalign})


#Plot 4: V100 Queries
algos = {"hfQuery":0, "cuQuery":1, "thSearch":2}
legends = {"hfQuery":"HashFight", \
           "cuQuery":"CUDPP", \
           "thSearch":"Thrust-Search"}

axes4 = canvas.cartesian(label = "V100 Querying, 900M Pairs",
                        xlabel = 'Load Factor',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid = (2,1,1),
                        margin = (30,50,50))
axes4.label.style = {"font-size":"12px"}
axes4.x.label.style = {"font-size":"8px"}
axes4.y.label.style = {"font-size":"8px"}
axes4.x.ticks.show = True
axes4.y.ticks.show = True
axes4.y.ticks.labels.angle = -45
axes4.x.ticks.far = "4px"
axes4.y.ticks.far = "4px"
axes4.y.label.location = "above"
axes4.y.label.offset = "30px"
axes4.x.ticks.locator = toyplot.locator.Explicit([1, 1.2, 1.4, 1.6, 1.8, 2.0])
axes4.y.ticks.locator = toyplot.locator.Explicit([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    x = df.numPairs.values
    y = numpy.array(data)
    plotstyle = GetStyle(toplot)
    label = legends.get(toplot)
    thisMarker = markers[algos.get(algo)]
    thisStroke = plotstyle.get("stroke")
    axes4.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes4._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "middle"
    #if "cu" in algo:
      #vertalign = "first-baseline"
    axes4.text(x[-1], y[-1], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"7.5px", \
                                          "-toyplot-vertical-align":vertalign})


#toyplot.pdf.render(canvas, "query.pdf")
toyplot.pdf.render(canvas, "vary-load-v100.pdf")
