import numpy
import pickle
import math

def average(x):
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    n = len(x)
    if n == 0:
        return -1
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    if xdiff2 == 0 or ydiff2 == 0: # Correlation undefined, here we take it as -1
        return -1
    return diffprod / math.sqrt(xdiff2 * ydiff2)

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    if sumxx*sumyy == 0:
        return 0
    return sumxy/math.sqrt(sumxx*sumyy)


##### Dataset size #####
#100,000 ratings (1-5) from 943 users on 1682 movies.
totuser,totitem = 943,1682

##### Filling the data in 2-D matrix #####
table = [[0 for x in range(totuser+1)] for y in range(totitem+1)]
avg_rating = [0 for x in range(totitem+1)] ## for storing the average of an item's ratings

u1b,u1t = open("ml-100k/u5.base","r").readlines(),open("ml-100k/u5.test","r").readlines()
for line in u1b:
	splitted = line.split()
	user,item,rating = int(splitted[0]),int(splitted[1]),int(splitted[2])
	table[item][user] = rating
for ii in range(1,totitem+1):
	totsum = sum(table[ii])
	totratings = totuser - (table[ii].count(0)-1)
	if totratings == 0:
		avg_rating[ii] = 0
	else:
		avg_rating[ii] = float(totsum)/totratings 

##### Creating Similarity Matrix for users ####

item_sim = [[0 for x in range(totitem+1)] for y in range(totitem+1)]
for u1 in range(1,totitem+1):
	for u2 in range(u1+1,totitem+1):
		print 'Progress => {0}\r'.format(u1),
		item1orig = table[u1]
		item2orig = table[u2]
		item1,item2 = [],[]
		for i in range(1,totuser+1):
			if table[u1][i] != 0 and table[u2][i] != 0: #only if both items are rated by a user
				item1.append(table[u1][i])
				item2.append(table[u2][i])
		curr_sim = pearson_def(item1,item2)
		curr_sim = float(1+curr_sim)/2
		#curr_sim = cosine_similarity(item1orig,item2orig)
		if math.isnan(curr_sim):
			item_sim[u1][u2],item_sim[u2][u1] = 0,0
		else:
			item_sim[u1][u2],item_sim[u2][u1] = curr_sim,curr_sim
#print item_sim
pickle.dump( item_sim, open( "item_u5.p", "wb" ) )


##### Testing and finding NMAE #######
item_sim = pickle.load( open( "item_u5.p", "rb" ) )
#print item_sim

totalerror = 0
testsetlen = len(u1t)
for line in u1t:
	splitted = line.split()
	user_s,item_s,rating = int(splitted[0]),int(splitted[1]),int(splitted[2])
	num,den = 0,0
	for i in range(1,totitem+1):
		if i!= item_s and table[i][user_s] != 0 and avg_rating[i] != 0:
			#num += table[i][user_s] * item_sim[i][item_s]
			num += (table[i][user_s]-avg_rating[i]) * item_sim[i][item_s]
			den += item_sim[i][item_s]
			#print table[i][user_s],item_sim[item_s][i],num,den

	if den == 0:
		print "cannot predict"
		testsetlen -= 1
		continue
	pred_rating = float(num)/den +avg_rating[item_s]
	pred_rating = int(round(pred_rating))
	totalerror += abs(rating - pred_rating)
	print "line=",line,"   orig rating = ",rating,"  predicted rating = ",pred_rating
mae = float(totalerror)/testsetlen
print "MAE = ",mae
print "Normalized MAE = ",mae/(5-1) # divided by the difference of ratings

"""
##### Predicting rating for a particular user and item ####
item_sim = pickle.load( open( "item_1_cosine.p", "rb" ) )
user_s,item_s = 1,2 # it is 5
num,den = 0,0
for i in range(1,totitem+1):
	if i!= item_s and table[i][user_s] != 0:
		num += table[i][user_s] * item_sim[item_s][i]
		den += item_sim[item_s][i]
		print table[i][user_s],item_sim[item_s][i],num,den
pred_rating = float(num)/den
print "predicted rating = ",pred_rating
"""

