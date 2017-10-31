import numpy
import pickle
import math

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def_sig(x, y,corated):
    assert len(x) == len(y)
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
    if n < corated :
        return (diffprod / math.sqrt(xdiff2 * ydiff2))*(float(n)/corated)
    return (diffprod / math.sqrt(xdiff2 * ydiff2))

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


##### Dataset size #####
#100,000 ratings (1-5) from 943 users on 1682 movies.
user,item = 943,1682

##### Filling the data in 2-D matrix #####
table = [[0 for x in range(item+1)] for y in range(user+1)]
avg_rating = [0 for x in range(user+1)] ## for storing the average of ratings
u1b,u1t = open("ml-100k/u1.base","r").readlines(),open("ml-100k/u1.test","r").readlines()
for line in u1b:
	splitted = line.split()
	user_s,item_s,rating = int(splitted[0]),int(splitted[1]),int(splitted[2])
	table[user_s][item_s] = rating 
for ii in range(1,user+1):
	totsum = sum(table[ii])
	totratings = item - (table[ii].count(0)-1)
	avg_rating[ii] = float(totsum)/totratings
#print "avg_rating ============= ",avg_rating
##### Creating Similarity Matrix for users ####

user_sim = [[0 for x in range(user+1)] for y in range(user+1)]
for u1 in range(1,user+1):
	for u2 in range(u1+1,user+1):
		print 'Progress => {0}\r'.format(u1),
		user1orig = table[u1]
		user2orig = table[u2]
		user1,user2 = [],[]
		for i in range(1,item+1):
			if table[u1][i] != 0 and table[u2][i] != 0: #only if both user have rated the item
				user1.append(table[u1][i])
				user2.append(table[u2][i])
		#curr_sim = numpy.corrcoef(user1,user2)
		#curr_sim = curr_sim[0][1]
		#print user1,user2
		curr_sim = pearson_def_sig(user1,user2,5)
		curr_sim = float(1+curr_sim)/2
		#curr_sim = cosine_similarity(user1orig,user2orig)
		if math.isnan(curr_sim):
			user_sim[u1][u2],user_sim[u2][u1] = 0,0
		else:
			user_sim[u1][u2],user_sim[u2][u1] = curr_sim,curr_sim
#print user_sim
pickle.dump( user_sim, open( "u1_sig5.p", "wb" ) )

##### Testing and finding NMAE #######
user_sim = pickle.load( open( "u1_sig5.p", "rb" ) )
totalerror = 0
testsetlen = len(u1t)
for line in u1t:
	splitted = line.split()
	user_s,item_s,rating = int(splitted[0]),int(splitted[1]),int(splitted[2])
	num,den = 0,0
	for i in range(1,user+1):
		if i!= user_s and table[i][item_s] != 0:
			#num += table[i][item_s] * user_sim[user_s][i]
			num += (table[i][item_s]-avg_rating[i]) * user_sim[user_s][i]
			den += user_sim[user_s][i]
			#print table[i][item_s],user_sim[user_s][i],num,den
	if den == 0:
		#print "cannot predict"
		testsetlen -= 1
		continue
	pred_rating = float(num)/den +avg_rating[user_s]
	pred_rating = int(round(pred_rating))
	totalerror += abs(rating - pred_rating)
	print "line=",line,"   orig rating = ",rating,"  predicted rating = ",pred_rating
mae = float(totalerror)/testsetlen
print "MAE = ",mae
print "Normalized MAE = ",mae/(5-1) # divided by the difference of ratings 

"""
##### Predicting rating for a particular user and item ####
user_sim = pickle.load( open( "user_sim_cosine.p", "rb" ) )
user_s,item_s = 1,6 # it is 5
num,den = 0,0
for i in range(1,user+1):
	if i!= user_s and table[i][item_s] != 0:
		num += table[i][item_s] * user_sim[user_s][i]
		den += user_sim[user_s][i]
		print table[i][item_s],user_sim[user_s][i],num,den
pred_rating = float(num)/den
print "predicted rating = ",pred_rating
"""

