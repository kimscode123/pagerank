import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    visit_probability = dict()

    linked_page = random.choice(page) # select a random page linked to by that page
    other_pages = set()

    for i in corpus:
        if i != page:
            other_pages.add(i) # add other pages

    link_from_other_page = random.choice(corpus[random.choice(other_pages)]) # gets a random page that is not page then gets a random link from other page

    visit_probability[linked_page] = damping_factor # random link is set to the damping factor
    visit_probability[link_from_other_page] = 1 - damping_factor # random link from other page is set to the opposite probability

    return visit_probability

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pagerank = dict()

    starting_page = random.choice(corpus) # should randomly choose a page from the corpus
    sampled_pages = transition_model(corpus, starting_page, damping_factor)

    count = 1
    while count <= n:
        for key, value in sampled_pages.items():
            pagerank[key] = value # Gets the key and value of the recently sampled page and adds it to the dictionary

        sampled_pages = transition_model(sampled_pages, sampled_pages[count], damping_factor) # returns a dictionary of multiple pages
        count += 1

    return pagerank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = dict()

    # start off by setting all pagerank to 1/N
    for page in corpus:
        pagerank[page] = 1/N # assumes all pages are equally likely to be visited

    difference = copy.deepcopy(pagerank) # variable to check if probability of pagerank changes by more than .001

    while True:
        count = 0 # counts the amount of page's pagerank that have an accuracy within .001

        for page in corpus:
            pagerank[page] = ((1 - damping_factor)/N) + (damping_factor * sum([i for i in page if i]/len(page)))
            difference[page] = difference[page] - pagerank[page]
        
        for page in corpus:
            if abs(difference[page]) <= .001:
                count += 1

            if count == N:
                break
            elif count > N:
                print("Count is greater than number of pages in corpus")
                break

        difference = copy.deepcopy(pagerank)

    return pagerank

if __name__ == "__main__":
    main()
