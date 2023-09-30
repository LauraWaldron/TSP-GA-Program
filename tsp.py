import sys
import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

# Define a city class to handle the cities
class City:
    def __init__(self, x, y):
        self.x = x #the x coordinate of the city
        self.y = y #the y coordinate of the city

    def distance(self, city): #define a function to compute the distance between cities
        xdist = abs(self.x - city.x) #find the x distance
        ydist = abs(self.y - city.y) #find the y distance
        distance = np.sqrt((xdist**2) + (ydist**2)) #use the pythagorean theorem to calculate the distance
        return distance #return the distance between cities

    def __repr__(self):  # Corrected '__rep__' to '__repr__'
        return "(" + str(self.x) + "," + str(self.y) + ")" #return the string with the cities inside it

# Define a fitness class to minimize the distance
#want to minimize route distance
class Fitness:
    def __init__(self, route):
        self.route = route #define the route
        self.distance = 0 #set the distance to 0
        self.fitness = 0.0 #set the fitness to 0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            #loop through all the cities to find the best fitness
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                #we need to start and end at the same place, so account for this here
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1] #add one city to the route
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance #return the new distance

    def routeFitness(self):  #define a function to handle the fitness of the route
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

 #create the initial population (1st generation)
def createRoute(cityList):
    #randomly select the order to visit each city (for one city)
    route = random.sample(cityList, len(cityList))
    return route

#Create a whole population, using the previous createRoute function
def initialPopulation(popSize, cityList):
    population = [] #set the population to an empty list
    #loop through all the cities and add them to the population
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population #return the completed population (aka cities)

#determine fitness of the functions (simulate survival of the fittest)
def rankRoutes(population):
    fitnessResults = {} #set fitnessResults to empty
    #loop through all of the functions and return their routes in an ordered fashion with the associated fitness score
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

#select the parents to create the next generation using Fitness porportionate selection
def selection(popRanked, eliteSize):
    #use the previous function to determine the best routes
    selectionResults = []
    #set up the roulette wheel by calculating the relative fitness weight for each individual
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"]) 
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    #introduce elitism
    for i in range(0, eliteSize):
        #use a for loop to add each item to the selectionResults list
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for idx in range(0, len(popRanked)):  # use a loop to iterate through our items
            if pick <= df.iat[idx, 3]:
                selectionResults.append(popRanked[idx][0])
                break
    return selectionResults

#use the previous results to create the mating pool
def matingPool(population, selectionResults):
    matingpool = [] #define a mating pool
    #iterate through selection results to find the best
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#create the next generation (breeding)
#use ordered crossover
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

#generalize the previous function to make the offspring population
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    #use the previous breed function to fill out the next generation
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

#use swap mutation so we don't swap any cities
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#extend the mutation function to the entire population
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#connect everything to return the next generation
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen) #rank the routes of the current generation
    selectionResults = selection(popRanked, eliteSize) #determine potential parents
    matingpool = matingPool(currentGen, selectionResults) #create the mating pool based on the selection
    children = breedPopulation(matingpool, eliteSize) #create the next generation
    nextGeneration = mutatePopulation(children, mutationRate) #apply mutation to the next generation
    return nextGeneration #return the next generation

#create the inital population, then loop through all the generations!!
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population) #create the initial population
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1])) #print the initial distance
    #iterate through all the generations and prepare the next generation
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    #print the final distance
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0] #set the best route
    bestRoute = pop[bestRouteIndex] #use the bestRouteIndex (found by rankRoutes)
    return bestRoute

#define a genetic algorithm graphing (not needed in this program, but here in case)
def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
   
#The main section of our program:
print("Welcome to the traveling salesman problem!")
genSize = int(input("Please enter the size of each generation: "))
genTotal = int(input("Please enter the total amount of generations: "))
cityNumber = int(input("Please enter the number of cities: "))
stagnant = int(input("Please enter the amount of stagnat generations we should stop after: "))
eliteSize = int(input("Please enter the float porportion: "))
mutationRate = float(input("Please enter the float mutation rate: "))

cityList = [] #set the city list to empty

#add the cities to the city list
for i in range(0, 25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

best_distance = float('inf') #find the best distance
#set the stagnat count to 0
stagnat_count = 0
#iterate through all the generations
for g in range(genTotal):
    #call geneticAlgorithm for each generation
    bestRoute = geneticAlgorithm(population = cityList, popSize = genSize, eliteSize=eliteSize, mutationRate=mutationRate, generations = 500)
    #find the current distance using rankRoutes
    current_distance = 1 /rankRoutes([bestRoute])[0][1] #what is this solving?
    #print the best solution from each generation
    print(f"Best solution from generation {g + 1} is {current_distance}")

    if current_distance < best_distance: #check if the distance is the best
        best_distance = current_distance #set the new best distance
        stagnant_count = 0 #reset the stagnant count
    else:
        stagnant_count += 1 #add one to the stagnant count
    
    #if the stagnant count is greater than what we set, stop the loop
    if stagnant_count >= stagnant:
        print(f"We had to stop after {stagnant} stagnant generations.")
        break

#print the results
print(f"The best solution is {best_distance} km.")
print("And it goes to the cities in this order: ")
print(bestRoute) #print the best route of the cities

#extra calls to cities prior to our changing of the algorthim
#geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
#geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

