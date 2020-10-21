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
title_0 = "Xeon Gold Insertions, 1.45B Pairs"
y_label_1 = "Query Throughput (1M pairs/sec)"
x_label_1 = "Load Factor"
title_1 = "Xeon Gold Querying, 1.45B Pairs"


df = pd.read_csv(filename, header=None, names=['numPairs', 'hfInsert', 'hfQuery', 'cuInsert', 'cuQuery', 'thSort', 'thSearch'])


df['hfInsert'] = 1450000000/df.hfInsert
df['cuInsert'] = 1450000000/df.cuInsert
df['thSort'] = 1450000000/df.thSort
df['hfQuery'] = 1450000000/df.hfQuery
df['cuQuery'] = 1450000000/df.cuQuery
df['thSearch'] = 1450000000/df.thSearch

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

canvas = toyplot.Canvas('3.5in', '6in')

#Plot 1: Xeon Gold Insertions
algos = {"hfInsert":0, "cuInsert":1, "thSort":2}
legends = {"hfInsert":"HashFight", \
           "cuInsert":"TBB-Map", \
           "thSort":"Thrust-Sort"}

axes = canvas.cartesian(label = title_0,
                        xlabel = 'Load Factor',
                        ylabel = 'Insert Throughput (1M pairs/sec)',
                        grid = (2,1,0),
                        margin = (40,50,40))
axes.label.style = {"font-size":"12px"}
axes.x.label.style = {"font-size":"8px"}
axes.y.label.style = {"font-size":"8px"}
axes.x.ticks.show = True
axes.y.ticks.show = True
axes.y.ticks.labels.angle = -90
axes.x.ticks.far = "4px"
axes.y.ticks.far = "4px"
axes.y.label.location = "above"
axes.y.label.offset = "30px"
axes.x.ticks.locator = toyplot.locator.Explicit([1, 1.2, 1.4, 1.6, 1.8, 2.0])
axes.y.ticks.locator = toyplot.locator.Explicit([0, 10, 100, 200, 300, 350])

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
    axes.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "middle"
    #if "cu" in algo:
      #vertalign = "first-baseline"
    axes.text(x[-1], y[-1], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"7.5px", \
                                          "-toyplot-vertical-align":vertalign})

#Plot 2: Xeon Gold Queries
algos = {"hfQuery":0, "cuQuery":1, "thSearch":2}
legends = {"hfQuery":"HashFight", \
           "cuQuery":"TBB-Map", \
           "thSearch":"Thrust-Search"}


axes2 = canvas.cartesian(label = title_1,
                        xlabel = 'Load Factor',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid=(2,1,1),
                        margin=(40,50,50))
axes2.label.style = {"font-size":"12px"}
axes2.x.label.style = {"font-size":"8px"}
axes2.y.label.style = {"font-size":"8px"}
axes2.x.ticks.show = True
axes2.y.ticks.show = True
axes2.y.ticks.labels.angle = -90
axes2.x.ticks.far = "4px"
axes2.y.ticks.far = "4px"
axes2.y.label.location = "above"
axes2.y.label.offset = "30px"
axes2.x.ticks.locator = toyplot.locator.Explicit([1, 1.2, 1.4, 1.6, 1.8, 2.0])
axes2.y.ticks.locator = toyplot.locator.Explicit([0, 25, 100, 200, 300, 400, 500])

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
    axes2.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes2._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "middle"
    #if "cu" in algo:
      #vertalign = "first-baseline"
    axes2.text(x[-1], y[-1], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"7.5px", \
                                          "-toyplot-vertical-align":vertalign})


#toyplot.pdf.render(canvas, "query.pdf")
toyplot.pdf.render(canvas, "vary-loads-xeon.pdf")
