from utils import structured_claude
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
import requests

load_dotenv()
HARDCOVER_AUTHORIZATION = os.getenv("HARDCOVER_AUTHORIZATION")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Building Effective AI Agents: demonstrating the routing workflow with a workflow that:
# - First determines whether a book was written by Hemingway, Shakespeare, or Vonnegut
# - Then, routes to either a special hemingway_llm, shakespeare_llm, or vonnegut_llm


# define structured outputs
class RouterResponse(BaseModel):
    category: Literal["Book", "Movie", "Artwork", "Other"] = Field(
        description="Category that this title belongs to"
    )


class ToolCall(BaseModel):
    name: str = Field(
        description="The first and last name of a person to use for the search tool call, e.g. Claude Monet"
    )


class FinalResponse(BaseModel):
    summary: str


# define tools
def search_books(author: str):
    """Searches Hardcover.app for the top 20 books written by an author"""

    # See Hardcover API: https://docs.hardcover.app/api/getting-started/
    url = "https://api.hardcover.app/v1/graphql"

    query = f"""
    query MyQuery {{
    authors(where: {{name: {{_eq: "{author}"}}}}) {{
        name
        contributions(
        order_by: {{book: {{users_count: desc_nulls_last}}}}, limit: 20
        ) {{
        book {{
            id
            title
            users_count
            description
        }}
        }}
    }}
    }}
    """

    json_payload = {"query": query, "variables": {}, "operationName": "MyQuery"}

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": HARDCOVER_AUTHORIZATION,
    }

    response = requests.post(url, json=json_payload, headers=headers)
    data = response.json()

    if "errors" in data:
        print("Errors:", data["errors"])
        return {"error": data["errors"]}
    else:
        print("Data:", data["data"])
        data = data["data"]
        if data.get("authors", []) and data["authors"][0].get("contributions", []):
            contributions = data["authors"][0]["contributions"]
            books = [item["book"] for item in contributions]
            return {"books": books}


def search_movies_by_director(director: str):
    """Searches TMDB for the top movies made by a director"""
    url = "https://api.themoviedb.org/3/search/person"
    params = {"query": director}

    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        director_info = data["results"][0]
        works = director_info["known_for"]
        movies = [
            {
                "title": work["title"],
                "overview": work["overview"],
                "release_date": work["release_date"],
            }
            for work in works
            if work["media_type"] == "movie"
        ]
        return {"movies": movies}
    else:
        return {"status_code": response.status_code, "error": response.text}


def search_artworks_by_artist(artist: str):
    """Searches the Art Institute of Chicago API for artworks made by an artist"""
    url = "https://api.artic.edu/api/v1/artworks/search"
    fields = [
        "title",
        "artist_title",
        "date_display",
        "description",
        "artwork_type_title",
    ]
    params = {
        "q": artist,
        "size": 20,
        "fields": ",".join(
            fields
        ),  # api expects fields as comma-separated, not &-separated (default for encoding lists)
    }

    headers = {
        "accept": "application/json",
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()

        artworks = data["data"]
        return {"artworks": artworks}
    else:
        return {"status_code": response.status_code, "error": response.text}


def routing(title: str):

    router_prompt = """
Determine if the title given refers to a Book, Movie, Artwork, or Other.
"""

    router_response = structured_claude(title, RouterResponse, system_prompt=router_prompt)
    print(f"Router response: {router_response}")
    if router_response.category == "Book":
        response = book_llm(title)
        return response
    elif router_response.category == "Movie":
        response = movie_llm(title)
        return response
    elif router_response.category == "Artwork":
        response = artwork_llm(title)
        return response
    else:
        return "This title doesn't belong to a Book, Movie, or Artwork!"


def book_llm(book_title: str) -> FinalResponse:
    tool_prompt = f"""
You are a literary expert!
Based on the title of the book {book_title}, find out what other books were also written by this author.
    """

    tools = [
        {
            "name": "search_books",
            "description": "Searches the Hardcover API for books written by an author",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The first and last name of the author with proper capitalization, e.g. Ernest Hemingway",
                    }
                },
                "required": ["name"],
            },
        }
    ]
    tool_call = structured_claude(tool_prompt, ToolCall, tools=tools)
    name = tool_call.name
    search_books_result = search_books(author=name)

    final_prompt = f"""
Given a list of books by {name}, write a short summary of their most well-known books.
Book list: {search_books_result}
    """

    response = structured_claude(final_prompt, FinalResponse)
    return response


def movie_llm(movie_title: str) -> FinalResponse:
    tool_prompt = f"""
You are a movie expert!
Based on the title of the movie {movie_title}, find out what other movies were also directed by this director.
    """

    tools = [
        {
            "name": "search_movies_by_director",
            "description": "Searches the Movie DB API for other works by this director",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The first and last name of the director with proper capitalization, e.g. Christopher Nolan",
                    }
                },
                "required": ["name"],
            },
        }
    ]

    tool_call = structured_claude(tool_prompt, ToolCall, tools=tools)
    name = tool_call.name
    search_movies_result = search_movies_by_director(director=name)

    final_prompt = f"""
Given a list of movies by {name}, write a short summary of their most well-known movies.
Movies list: {search_movies_result}
    """

    response = structured_claude(final_prompt, FinalResponse)
    return response


def artwork_llm(artwork: str) -> FinalResponse:
    tool_prompt = f"""
You are an art expert!
Based on the title of an artwork {artwork}, find out what other artworks were also made by this artist.
    """

    tools = [
        {
            "name": "search_artworks_by_artist",
            "description": "Searches the  Art Institute of Chicago API for other works by this director",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The first and last name of the artist with proper capitalization, e.g. Claude Monet",
                    }
                },
                "required": ["name"],
            },
        }
    ]

    tool_call = structured_claude(tool_prompt, ToolCall, tools=tools)
    name = tool_call.name
    search_artworks_result = search_artworks_by_artist(artist=name)

    final_prompt = f"""
Given a list of artworks by {name}, write a short summary of their most well-known artworks.
Artworks list: {search_artworks_result}
    """

    response = structured_claude(final_prompt, FinalResponse)
    return response

res = routing("Mona Lisa")
if res.summary:
    summary = res.summary
    print(f"Final summary: {res.summary}")
else:
    print(res)