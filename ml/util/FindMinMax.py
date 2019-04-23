
import sys
import os
import time

import numpy as np


class FindMinMax:
    
    def __init__(self):
        self.parse_args(sys.argv)
        
    def parse_args(self, argv):
        """parse args and set up instance variables"""
        try:
            self.working_dir = os.getcwd()
        except:
            print(self.usage())
            sys.exit(1)
    
    def cal(self, value1, value2):
        
        start_time = time.time()
        
        min_value1 = min(value1)
        min_index = np.argmin(value1)
        min_value2 = value2[min_index]
        
        print("Min index: " + str(min_index) + ", Min value: " + str(min_value1) + ", " + str(min_value2))
        
        end_time = time.time()   
        
        print("Cost : " + str(end_time - start_time) + " seconds")
        
    def usage(self):
        return """
        Train a Logistic Regression with libsvm file
        Usage:
            $ python array_info.py
        """

if __name__ == "__main__":
    value1 = [9.99726912598714, 9.992260653560448, 9.987781339760517, 9.983463552236774, 9.97923354361373, 9.9750409281286, 9.970962757629824, 9.967014252686347, 9.963129253474932, 9.959325029170989, 9.955626014595026, 9.951965260245315, 9.948351535600622, 9.944807156709054, 9.94139692102124, 9.938122175625244, 9.934941737512537, 9.93178384288089, 9.928658741283689, 9.92563619673184, 9.922671281750874, 9.919722347278045, 9.916866770099741, 9.91404426594791, 9.911268598371041, 9.908540666033376, 9.90585909611614, 9.903214766401327, 9.900609053150358, 9.897992975381309, 9.895428338875968, 9.892980624112445, 9.890614709164502, 9.88826989591086, 9.885928409081199, 9.883602375125003, 9.881312628188235, 9.879033349315115, 9.876760427056798, 9.87452170306431, 9.872331521579992, 9.87020375585475, 9.868111216246037, 9.866035306421175, 9.863958835584627, 9.8619636811517, 9.860002743001521, 9.858085869906208, 9.856243557544602, 9.854401350726073, 9.85257695930952, 9.850747481495075, 9.84890669275315, 9.847091279307046, 9.84531179854047, 9.843525444856297, 9.841761738038171, 9.840037227227935, 9.838341127237765, 9.836680348492356, 9.835030530689549, 9.833350797760795, 9.831635223672597, 9.82993893187148, 9.828301353791069, 9.826694138599866, 9.825078866109097, 9.823453705226154, 9.821876307121912, 9.820354880781984, 9.818901446993609, 9.81746901535255, 9.816071070952814, 9.814712492538929, 9.813331871873357, 9.811954671555212, 9.810630825226788, 9.809359517933842, 9.808106506539783, 9.806873507709998, 9.805671407233461, 9.804491451641432, 9.80330739578012, 9.802094184396001, 9.800911568206235, 9.799758809415438, 9.798632102407348, 9.797541508114678, 9.796455022655802, 9.795381109696445, 9.7942923015988, 9.793194564425699, 9.79213426013448, 9.79110104592441, 9.79010849079926, 9.789110592811427, 9.788114820454728, 9.787114467020494, 9.786147520569505, 9.78518995294154]
    
    value2 = [359001, 691913, 1036230, 1393758, 1752256, 2094305, 2447039, 2774199, 3125952, 3462245, 3772704, 4096820, 4424935, 4772110, 5114099, 5455718, 5797803, 6132765, 6465353, 6801558, 7133991, 7458915, 7787146, 8130363, 8471483, 8819107, 9144925, 9501642, 9833142, 10156041, 10470336, 10806686, 11132681, 11459449, 11780779, 12115124, 12445387, 12793985, 13141842, 13494917, 13819324, 14150149, 14521943, 14869785, 15212040, 15547581, 15885776, 16234712, 16571674, 16915419, 17257244, 17601464, 17934700, 18250247, 18594723, 18920995, 19251830, 19602425, 19953436, 20289349, 20635938, 20952401, 21299109, 21635485, 21975267, 22314548, 22644850, 22965740, 23293382, 23622848, 23985823, 24312753, 24640461, 24995184, 25338889, 25650545, 25979935, 26328815, 26667266, 26998974, 27331161, 27666479, 28009667, 28349603, 28689904, 29016450, 29339835, 29660342, 29985773, 30344183, 30662629, 31018187, 31367031, 31705323, 32055862, 32413639, 32755771, 33084679, 33399089, 33741688]
    
    runner = FindMinMax()
    runner.cal(value1, value2)
        



