# -*- coding: utf-8 -*-
"""
Created on a cloudy day

@author: Jamie Baggott
@id: R00149982
@Cohort: SD3A
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("movie_metadata.csv", encoding='utf8')


def Task1():
    df["color"] = df["color"].str.strip()  # Strips the extra space in the column

    group_actors = df.groupby("color")["actor_1_name"].value_counts().to_frame(
        "movies").reset_index()  # Groups the actors together

    bw_actors = group_actors[
        (group_actors["color"] == "Black and White")]  # Groups the actors that have appeared in Black and White movies
    multiple_bw_actors = bw_actors[
        (bw_actors["movies"] > 1)]  # Takes that group and see who has been in more than 1 Black and White movie

    actors = multiple_bw_actors["actor_1_name"].tolist()  # Adds these actors to a list

    for actor in actors:
        print("Name: " + actor)
    return actors


def Task2():
    foreign = df[(df["language"] != "English")]  # Finds the foreign language films
    over_150_mins = foreign[(foreign[
                                 "duration"] > 150)]  # Finds within that section of foreign films the ones that last
    # longer than 150 minutes

    countries = over_150_mins[
        "country"].unique().tolist()  # Takes the unique countries so as not to duplicate information as many
    # countries could have many films longer than 150 mins

    for country in countries:
        print("Country: " + country)
    return countries


def Task3():
    df["gross"].fillna(df["gross"].mean(),
                       inplace=True)  # Fills the blank spaces that are missing the gross with the average gross of
    # the file
    movie_income_gained = df.groupby("title_year")["gross"].sum()  # Groups the earnings by year

    plt.figure(figsize=(10, 10))

    plt.title("MOVIE INCOME OVER THE LAST CENTURY", fontsize=15)
    plt.xlabel("Decade", fontsize=10)
    plt.ylabel("Earnings", fontsize=10)

    x_ticks = np.arange(1900, 2030, 10)  # Sets the range that years will go through
    plt.xticks(x_ticks)

    y_ticks = np.arange(0, 20000000000, 1000000000)  # Sets the range that the earnings will go through
    plt.yticks(y_ticks)

    plt.plot(movie_income_gained,
             label="Income gained for movies in last 100 years. Earnings displayed in Billions")  # Plots the income
    # and years for the earnings
    plt.legend()  # Shows the legend
    plt.show()  # Shows the information
    return None


def Task4():
    missing_value = float("NaN")  # Sets the missing value
    df.replace("", missing_value, inplace=True)
    df.dropna(subset=["gross"], inplace=True)  # Deletes the information with no information in it

    plt.figure(figsize=(18, 10))

    movie_year = df[df["title_year"] >= 1990]  # Looks for movies that are beyond 1989
    years = sorted(movie_year["title_year"].unique())  # Takes the unique years
    percentage_double_budget = []

    for year in years:
        movies = movie_year[movie_year["title_year"] == year]
        movies_that_doubled = 0

        for budget, gross in zip(movies["budget"], movies["gross"]):
            doubled_budget = (budget * 2)  # Sets the doubled budget variable
            if gross >= doubled_budget:
                movies_that_doubled = (movies_that_doubled + 1)

        percentage_double_budget.append(movies_that_doubled / len(
            movies))  # Gets the percentage by having the profitable movies divided by total number of movies

    plt.title("PERCENTAGE OF MOVIES THAT GROSSED DOUBLE THEIR BUDGET IN REVENUE", fontsize=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Percentage", fontsize=10)

    x_ticks = np.arange(1988, 2020, 2)
    plt.xticks(x_ticks)

    y_ticks = np.arange(0, 1, .05)
    plt.yticks(y_ticks)

    plt.plot(years, percentage_double_budget, label="Percentage of films that doubled their budget in revenue")
    plt.legend()
    plt.show()


def Task5():
    missing_value = float("NaN")  # Gets rid of the missing value for countries
    df.replace("", missing_value, inplace=True)
    df.dropna(subset=["country"], inplace=True)

    c30 = df.country.value_counts()  # Finds the time each country appears
    countries_over30 = (c30[c30 >= 30])  # Finds only the countries that appear 30 or more times

    countries = list(df["country"].unique())  # Makes a list of countries

    countries = [ele for ele in countries if ele not in ["USA", "UK"]]  # Excludes the UK and USA from the list
    countries = [ele for ele in countries if
                 ele in countries_over30]  # Only includes the countries that appear in the countries_over30 variable

    total_movies = []
    for country in countries:
        total_movies.append(len(df[df["country"] == country]))  # Adds each country and it's films to the array of films

    percent_movies = [item * 100 / len(df) for item in
                      total_movies]  # Gets the percentage of the whole set that these movies make up

    plt.figure(figsize=(15, 10))

    plt.title("NUMBER OF MOVIES OUTSIDE OF THE USA AND UK", fontsize=15)
    plt.xlabel("Country", fontsize=10)
    plt.ylabel("Number of Movies", fontsize=10)

    x_ticks = np.arange(0, 8, 1)
    plt.xticks(x_ticks, rotation=45)

    y_ticks = np.arange(0, 170, 10)
    plt.yticks(y_ticks)

    plt.plot(countries, total_movies)
    plt.legend(["Number of movies of countries with over 30 films excluding USA and UK"])
    plt.show()

    plt.figure(figsize=(15, 10))

    x_ticks = np.arange(0, 8, 1)
    plt.xticks(x_ticks, rotation=45)

    y_ticks = np.arange(0, 5, .25)
    plt.yticks(y_ticks)

    plt.title("PERCENTAGE OF SHARE THAT THESE COUNTRY'S FILMS TAKE UP", fontsize=15)
    plt.xlabel("Country", fontsize=10)
    plt.ylabel("Percentage", fontsize=10)

    plt.plot(countries, percent_movies)
    plt.legend(["Percentage of where movies are made without the UK or USA"])
    plt.show()


def Task6():
    missing_value = float("NaN")
    df.replace("", missing_value, inplace=True)
    df.dropna(subset=["duration"], inplace=True)

    plt.figure(figsize=(10, 10))

    x_ticks = np.arange(0, 540, 30)
    plt.xticks(x_ticks)

    y_ticks = np.arange(0, 850, 50)
    plt.yticks(y_ticks)

    plt.title("DURATION OF FILMS", fontsize=15)
    plt.xlabel("Time in Minutes", fontsize=10)
    plt.ylabel("Number of Movies", fontsize=10)

    times = df["duration"]  # Sets the times of each of the films

    plt.hist(times, bins=90,
             label="Number of films and their run times")  # Makes a histogram displaying the details of how long the
    # durations of films are
    plt.legend()
    plt.show()


Task1()
Task2()
Task3()
Task4()
Task5()
Task6()
