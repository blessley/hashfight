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

canvas = toyplot.Canvas('6.5in', '6in')

#Plot 1: K40 Insertions
algos = {"hfInsert2":0, "cuInsert2":1, "thSort2":2}
legends = {"hfInsert2":"HashFight", \
           "cuInsert2":"CUDPP", \
           "thSort2":"Thrust-Sort"}

axes = canvas.cartesian(label = title_0,
                        xlabel = 'Load Factor',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid = (2,2,0),
                        margin = (40,50,30))
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
axes.y.ticks.locator = toyplot.locator.Explicit([0, 100, 200, 300, 400, 500, 600, 700])

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
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})

#Plot 2: K40 Queries
algos = {"hfQuery2":0, "cuQuery2":1, "thSearch2":2}
legends = {"hfQuery2":"HashFight", \
           "cuQuery2":"CUDPP", \
           "thSearch2":"Thrust-Search"}


axes2 = canvas.cartesian(label = title_1,
                        xlabel = 'Load Factor',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid=(2,2,1),
                        margin=(40,50,30))
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
#axes2.y.ticks.locator = toyplot.locator.Explicit([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

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
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})


#Plot 3: V100 Insertions
algos = {"hfInsert":0, "cuInsert":1, "thSort":2}
legends = {"hfInsert":"HashFight", \
           "cuInsert":"CUDPP", \
           "thSort":"Thrust-Sort"}

axes3 = canvas.cartesian(label = "V100 Insertions, 900M Pairs",
                        xlabel = 'Load Factor',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid = (2,2,2))
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
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})


#Plot 4: V100 Queries
algos = {"hfQuery":0, "cuQuery":1, "thSearch":2}
legends = {"hfQuery":"HashFight", \
           "cuQuery":"CUDPP", \
           "thSearch":"Thrust-Search"}

axes4 = canvas.cartesian(label = "V100 Querying, 900M Pairs",
                        xlabel = 'Load Factor',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid = (2,2,3))
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
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})


#toyplot.pdf.render(canvas, "query.pdf")
toyplot.pdf.render(canvas, "vary-load.pdf")
