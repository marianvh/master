from math import ceil, floor, sin, asin, cos, sqrt
import numpy as np
from import_data import save, load
from sklearn.metrics.pairwise import haversine_distances
from datetime import date
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import numba
#https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        

class Helpers:

    def __init__(self, trajectories, num_of_sheep=1, start_sheep=0, changed=False):
        self.hours_dist_matrix = load("hour_dist_matrix")
        if(changed):
            self.dist_dist_matrix = self.create_dist_matrix(trajectories, num_of_sheep, start_sheep)
        else:
            self.dist_dist_matrix = load("distance_matrix")
        return
    # returns distance between two coordinates in meters :)
    def haversine_distance(self, lat1, lon1, lat2, lon2 ):
        try:
            #Calculate the great circle distance in meters between two points 
            #on the earth (specified in decimal degrees)
            # convert decimal degrees to radians 
            #lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

            # haversine formula 
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a)) 
            r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
            return c * r * 1000
        except:
            print("error in haversine params")


    def create_dist_matrix(self, trajectories, num_of_sheep=1, start_sheep=0):
            ids = trajectories.id.unique()
            newids = []
            for i in ids:
                newids.append(str(trajectories[trajectories.id == i].iloc[0].datetime.year)+"_"+i)
            newids.sort()
            
            sheep_to_use = list(map(lambda i: i.split('_',1)[1],newids))[start_sheep:start_sheep+num_of_sheep]
            data = trajectories[trajectories.id.isin(sheep_to_use)]
            data[['latitude', 'longitude']] = np.deg2rad(data[['latitude', 'longitude']])
            haversine_dist_matrix = haversine_distances(data[['latitude', 'longitude']])*6371000
            rows, cols = haversine_dist_matrix.shape
            print(rows, cols)
            scaler = QuantileTransformer()
            reshaped = haversine_dist_matrix.reshape(-1,1)
            scaled = scaler.fit_transform(reshaped)
            data = scaled.reshape(rows, cols)
            save("distance_matrix", data)
            return data
        
    def velocity(self,distance, timestart, timestop):
        try:
            duration = timestop - timestart
            duration_in_s = duration.total_seconds()
            if(duration_in_s <= 0):
                print("duration is zero ")
                print(f"starttime: {timestart}, endtime: {timestop}, distance: {distance}",)
                return 0

            #11.176 meter per sekund er maxgrense for sauer
            return distance/duration_in_s
        except ZeroDivisionError as ze:
            print(ze)
            print("end time - start time is probably zero. ")
        except:
            print("error in velocity params")

    def velocitytudes(self,starttraj, endtraj):
        try:
            latdisplacement = endtraj[1]- starttraj[1]
            longdisplacement = endtraj[2]-starttraj[2]
            duration = endtraj[0] - starttraj[0]
            duration_in_s = duration.total_seconds()
            distance = sqrt(latdisplacement**2 + longdisplacement**2)
            return distance/duration_in_s
        except:
            print("error in velocity params")

    #returns rotation in radians
    #LITT usikker på denne utregningen ass
    def rotation(self,lat1, lon1, lat2, lon2 ):
        try:
            # https://stackoverflow.com/questions/15231466/whats-the-difference-between-pi-and-m-pi-in-objc
            # regner ut endring i rotasjon mellom punktene: blir det samme som retningen på linja
            dy = lat2 - lat1
            dx = cos(np.pi/180*lat1)*(lon2 - lon1)
            angle = np.arctan2(dy, dx)
            return angle
        except:
            print("error in rotation params")

    def timeToNumber(self,datet):
        time = datet.time()
        #From this we get 48 possible hour values 0 ~ 47
        hourlycycle = ((time.hour % 24)*2 + (1 if time.minute > 30 else 0))
        seasonstart = date(datet.year, 6, 1)
        days = (datet.date() - seasonstart).days
    
        return days, hourlycycle

    def distvelrot(self,lat1, lon1, lat2, lon2, timestart, timestop):
        
        dist = self.haversine_distance(lat1, lon1, lat2, lon2)
        vel = self.velocity(dist, timestart, timestop)
        rot = self.rotation(lat1, lon1, lat2, lon2)
        days, hours = self.timeToNumber(timestart)
        return  dist, days, int(hours), vel, rot


    #penalty is how much penalty should be given per period. penaltyperiod defines how many days a period is.
    # If penaltyperiod is 7 then each week the distancevalues for that week are added a penalty to represent the passage of time
    # If penaltyperiod is 1 then each 24 hours a new penalty is added.  
    def just_haversine(self,X):
        arr = haversine_distances(np.deg2rad(X[:,:2]))* 6371000
        return arr
        
# first try at some distance method
    def distance_self(self,X, penalty=70, penaltyperiod_days=7):
        arr = haversine_distances(np.deg2rad(X[:,:2]))* 6371000
        q = len(arr)
        palindrome_length = -1+q*2
        # SAY We add 70 meters as penalty per week, so week 1 is 0 m, week 2 is 70 meters etc ...
        # # that is we add 70 for every 48*7 point. 
        penalty_list = [x*penalty for x in range(ceil(q/(48*penaltyperiod_days)))]
        penalty_matrix = np.repeat(penalty_list, 48*penaltyperiod_days)[:q]
        palindrome = [*list(reversed(penalty_matrix)),*penalty_matrix[1:]]
        
        palindrome_middle = floor(len(palindrome)/2)
        new_distances = np.array(list(reversed(np.lib.stride_tricks.sliding_window_view(palindrome,palindrome_middle+1))))
        neww = arr + new_distances
        return neww


    # 48x48 matrix where row, column is the distance in hours between row, column
    def hour_dist(self,):
        mid = ceil(47/2)
        halflist =np.arange(mid+1)
        row = [[*halflist, *list(reversed(halflist[1:-1]))]]
        matrix = np.repeat(row, 49, axis=0)
        @numba.njit
        
        def shift_numba(x, r=48):
            for i in range(1, r):
                x[i] = np.append(x[i,r-i:], x[i, :r-i])
            return x 

        m = shift_numba(matrix, len(row[0]))
        scaler = MinMaxScaler()
        scaler.fit(m)
        m = scaler.transform(m)
        save("hour_dist_matrix", m)
        return m
        
    #two rows T1 and T2 !
    # ["latitude", "longitude", "distance", "days", "hours", "velocity", "rotation", "altitude"]


    def custom_metric(self, T1, T2):
        lat1, lon1, dist1, days1, hours1, vel1, rot1, alt1, index1 = T1
        lat2, lon2, dist2, days2, hours2, vel2, rot2, alt2, index2 = T2
        phys_dist = self.dist_dist_matrix[int(index1), int(index2)]  
        dist_next = (dist1-dist2) 
        days_dist = (days2-days1) 
        vel_dist = (vel2-vel1) 
        rot_dist = (rot2-rot1) 
        alt_dist = alt2-alt1
        hour_dist = self.hours_dist_matrix[int(hours1), int(hours2)]
        dist = sqrt(phys_dist**2 + days_dist**2 + rot_dist**2 + vel_dist**2 + dist_next**2 + alt_dist**2 + hour_dist**2)
        
        return dist

