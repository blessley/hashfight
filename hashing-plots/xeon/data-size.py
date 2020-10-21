# coding: utf-8

# In[19]:

import numpy as np
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
title_0 = "K40 Insertions"
y_label_1 = "Query Throughput (billions pairs/sec)"
x_label_1 = "Load Factor"
title_1 = "K40 Querying"


df = pd.read_csv(filename, header=None, names=['numPairs0', 'hfInsert00', 'hfQuery00', 'cuInsert00', 'cuQuery00', 'thSort0', 'thSearch0', 'hfInsert01', 'hfQuery01', 'cuInsert01', 'cuQuery01', 'numPairs1', 'hfInsert10', 'hfQuery10', 'cuInsert10', 'cuQuery10', 'thSort1', 'thSearch1', 'hfInsert11', 'hfQuery11', 'cuInsert11', 'cuQuery11'])


df['hfInsert00'] = df.numPairs0/df.hfInsert00
df['cuInsert00'] = df.numPairs0/df.cuInsert00
df['thSort0'] = df.numPairs0/df.thSort0
df['hfInsert01'] = df.numPairs0/df.hfInsert01
df['cuInsert01'] = df.numPairs0/df.cuInsert01
df['hfQuery00'] = df.numPairs0/df.hfQuery00
df['cuQuery00'] = df.numPairs0/df.cuQuery00
df['thSearch0'] = df.numPairs0/df.thSearch0
df['hfQuery01'] = df.numPairs0/df.hfQuery01
df['cuQuery01'] = df.numPairs0/df.cuQuery01

df['hfInsert10'] = df.numPairs1/df.hfInsert10
df['cuInsert10'] = df.numPairs1/df.cuInsert10
df['thSort1'] = df.numPairs1/df.thSort1
df['hfInsert11'] = df.numPairs1/df.hfInsert11
df['cuInsert11'] = df.numPairs1/df.cuInsert11
df['hfQuery10'] = df.numPairs1/df.hfQuery10
df['cuQuery10'] = df.numPairs1/df.cuQuery10
df['thSearch1'] = df.numPairs1/df.thSearch1
df['hfQuery11'] = df.numPairs1/df.hfQuery11
df['cuQuery11'] = df.numPairs1/df.cuQuery11

df['numPairs0'] = df.numPairs0/1000000
df['numPairs1'] = df.numPairs1/1000000

colors = ["green", "steelblue", "darkred",
          "green", "steelblue"]
markers = ["o", "--", "^",
           "o", "--"]
exts = ["", "2"]

def GetStyle(toplot) :
  style = {}
  keys = algos.keys()
  for key in keys:
    if key in toplot :
      index = algos.get(key)
      style["stroke"] = colors[index]
    if "01" in toplot or "11" in toplot :
      style["stroke-dasharray"] = "2, 2"
  return style


canvas = toyplot.Canvas('6.5in', '6in')

#Plot 1: K40 Insertions
algos = {"hfInsert10":0, "cuInsert10":1, "thSort1":2,
         "hfInsert11":3, "cuInsert11":4}
legends = {"hfInsert10":"HashFight,1.03", \
           "cuInsert10":"CUDPP,1.03", \
           "thSort1":"Thrust-Sort", \
           "hfInsert11":"HashFight,1.50", \
           "cuInsert11":"CUDPP,1.50"}

axes = canvas.cartesian(label = title_0,
                        xlabel = 'Key-Value Pairs (1M)',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid = (2,2,0),
                        margin = (40,50,30))
axes.label.style = {"font-size":"12px"}
#axes.label.location = "above"
#axes.label.offset = "-20px"
axes.x.label.style = {"font-size":"8px"}
axes.y.label.style = {"font-size":"8px"}
axes.x.ticks.show = True
axes.y.ticks.show = True
axes.y.ticks.labels.angle = -90
axes.x.ticks.far = "4px"
axes.y.ticks.far = "4px"
axes.y.label.location = "above"
axes.y.label.offset = "30px"
axes.x.ticks.locator = toyplot.locator.Explicit([50, 100, 200, 300, 400, 500])
axes.y.ticks.locator = toyplot.locator.Explicit([0, 100, 200, 300, 400, 500, 600])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    t = df.numPairs1.values
    m = t.argmax()
    x = [c for c in itertools.islice(t, 0, m)]
    t = numpy.array(data)
    y = [c for c in itertools.islice(t, 0, m)]
    plotstyle = GetStyle(toplot)
    label = legends.get(toplot)
    thisMarker = markers[algos.get(algo)]
    thisStroke = plotstyle.get("stroke")
    print len(x)
    print len(y)
    axes.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "middle"
    #if "cu" in algo:
      #vertalign = "first-baseline"
    y_end = np.count_nonzero(~np.isnan(y)) - 1
    axes.text(x[y_end], y[y_end], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})

#Plot 2: K40 Queries
algos = {"hfQuery10":0, "cuQuery10":1, "thSearch1":2,
         "hfQuery11":3, "cuQuery11":4}
legends = {"hfQuery10":"HashFight,1.03", \
           "cuQuery10":"CUDPP,1.03", \
           "thSearch1":"Thrust-Search", \
           "hfQuery11":"HashFight,1.50", \
           "cuQuery11":"CUDPP,1.50"}

axes2 = canvas.cartesian(label = title_1,
                        xlabel = 'Key-Value Pairs (1M)',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid=(2,2,1),
                        margin = (40,50,30))

#axes2.label.location = "above"
#axes2.label.offset = "-20px"
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

axes2.x.ticks.locator = toyplot.locator.Explicit([50, 100, 200, 300, 400, 500])
axes2.y.ticks.locator = toyplot.locator.Explicit([0, 100, 200, 300, 400, 500])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    t = df.numPairs1.values
    m = t.argmax()
    x = [c for c in itertools.islice(t, 0, m)]
    t = numpy.array(data)
    y = [c for c in itertools.islice(t, 0, m)]
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

    y_end = np.count_nonzero(~np.isnan(y)) - 1
    axes2.text(x[y_end], y[y_end], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})


#Plot 3: V100 Insertions
algos = {"hfInsert00":0, "cuInsert00":1, "thSort0":2,
         "hfInsert01":3, "cuInsert01":4}
legends = {"hfInsert00":"HashFight,1.03", \
           "cuInsert00":"CUDPP,1.03", \
           "thSort0":"Thrust-Sort", \
           "hfInsert01":"HashFight,1.50", \
           "cuInsert01":"CUDPP,1.50"}

axes3 = canvas.cartesian(label = "V100 Insertions",
                        xlabel = 'Key-Value Pairs (1M)',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid = (2,2,2),
                        margin = (30,50,50))

axes3.label.location = "above"
axes3.label.offset = "-20px"
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

axes3.y.domain.max = 3500
axes3.x.ticks.locator = toyplot.locator.Explicit([50, 500, 1000, 1450])
axes3.y.ticks.locator = toyplot.locator.Explicit([0, 500, 1000, 1500, 2000, 2500, 3000])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    x = df.numPairs0.values
    y = numpy.array(data)
    print "plot 3:\n"
    print x
    print y
    plotstyle = GetStyle(toplot)
    label = legends.get(toplot)
    thisMarker = markers[algos.get(algo)]
    thisStroke = plotstyle.get("stroke")
    axes3.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes3._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "middle"
    if "hfInsert01" in algo:
      vertalign = "first-baseline"
    
    y_end = np.count_nonzero(~np.isnan(y)) - 1
    axes3.text(x[y_end], y[y_end], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})


#Plot 4: V100 Queries
algos = {"hfQuery00":0, "cuQuery00":1, "thSearch0":2,
         "hfQuery01":3, "cuQuery01":4}
legends = {"hfQuery00":"HashFight,1.03", \
           "cuQuery00":"CUDPP,1.03", \
           "thSearch0":"Thrust-Search", \
           "hfQuery01":"HashFight,1.50", \
           "cuQuery01":"CUDPP,1.50"}

axes4 = canvas.cartesian(label = "V100 Querying",
                        xlabel = 'Key-Value Pairs (1M)',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid = (2,2,3),
                        margin = (30,50,50))

axes4.label.location = "above"
axes4.label.offset = "-20px"
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

axes4.y.domain.max = 4000
axes4.x.ticks.locator = toyplot.locator.Explicit([50, 500, 1000, 1500])
axes4.y.ticks.locator = toyplot.locator.Explicit([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    x = df.numPairs0.values
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

    y_end = np.count_nonzero(~np.isnan(y)) - 1
    axes4.text(x[y_end], y[y_end], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})

canvas.aspect = 'fit-range'
toyplot.pdf.render(canvas, "data-size-2.pdf")
