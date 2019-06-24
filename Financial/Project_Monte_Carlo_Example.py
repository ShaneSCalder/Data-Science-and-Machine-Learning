#!/usr/bin/env python
# coding: utf-8

# <h1>Monte Carlo Simulation to determine Project Contingency</h1>
# <ul>
#     <li>Projected Costs</li>
#     <li>Schedule</li>
#     <li>Iterations</li>
#     <li>Contingency</li>
# </ul>

# In[69]:


# Import Libraries
import numpy as np
import matplotlib.pyplot as plt


# In[70]:


#Project Projected Costs

"""
Project costs based upon Work Breakdown Schedule (WBS) i.e:
1.0 Conceptual Engineering
2.0 Detailed Engineering
3.0 Procurement
4.0 Construction
5.0 Starup
"""
#In millions 
projected_cost = 120

#Schedule Developement 
"""
One example of project schedule levels (AACE, PMI Client organizations and EPC's may 
have different definitions) Some models have level 1 as detailed and 4 - 5 as 
pre-feasability study.

Level 1 & 2 pre-feasibility studies 
Level 3 High level detail Starting CPM (Critical Path method)
It should include major elements of design, engineering, procurement, construction,
testing, commissioning and/or start-up
Level 4 Execution Schedule, also called a Project Working Level Schedule.
Level 5 Detailed Schedule.

Standard deviation 
Level 1 & 2 %40 to %80 cost overrun (some industries this can be as high as 200%)
Recommend add 40% and run 40% Standard deviation 
Level 3 20% 
Level 4 10%
Level 5 5%

"""
# Use information to develope stdev example will use a level 3 schedule
schedule_stdev = 20

#Iterations number of simulated costs (use 500 as a minimum)
iterations = 1000


# In[71]:


project = np.random.normal(projected_cost, schedule_stdev, iterations)
project


# In[72]:


plt.figure(figsize = (15,5))
plt.plot(project)


# In[86]:


#Construction costs to mechanical completion 70% of cost with a standard deviation of 20%
#you can break down costs indetail, risk etc. In this section

construction_costs = - (project * np.random.normal(.7,0.2))
other_costs = - (project * np.random.normal(.3,0.2))
# Use risk register for project amount (if risks are positive use + sign)
identified_risks = - (project * np.random.normal(.03,0.1))

plt.figure(figsize=(15, 6))
plt.plot(construction_costs)
plt.title('Construction Costs')
plt.show()

plt.figure(figsize=(15, 6))
plt.title('Other Costs')
plt.plot(other_costs)
plt.show()

plt.figure(figsize=(15, 6))
plt.title('Identified Risks')
plt.plot(identified_risks)
plt.show()


# <h2>Mean and Standard Deviation</h2>

# In[87]:


construction_costs.mean()


# In[88]:


construction_costs.std()


# In[89]:


other_costs.mean()


# In[90]:


other_costs.std()


# In[91]:


identified_risks.mean()


# In[92]:


identified_risks.std()


# <h2>Contingency</h2>

# In[93]:


contingency = project + (construction_costs + other_costs + identified_risks)
contingency

plt.figure(figsize=(15,6))
plt.plot(contingency)
plt.show()


# In[94]:


max(contingency)


# In[95]:


min(contingency)


# In[96]:


contingency.mean()


# In[97]:


contingency.std()


# In[101]:


plt.figure(figsize =(10,6))
plt.title('Project Contingency')
plt.xlabel('Contingency Value')
plt.ylabel('Project Costs')
plt.hist(contingency, bins =50);
plt.show()


# <h2>Recomendations</h2>

# <h3> Project Exposure</h3>
# <p> The project could expose the organization to a maximum project overrun of 29.34 million dollars.</p>

# <h3>Contingency Recomendation</h3>
# <p>Based upon the Monte Carlo simulation a recommendation of 18.46 million contingency be set up for this project.</p>

# In[ ]:




