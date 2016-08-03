# Jekyll Themed Blog 

To launch the website, click [`here`](https://tsmith5151.github.io/Blog/).

#### How to run locally

Before getting stated, you will need to install jekyll and then follow the following steps.

1. Open a terminal 
2. Create the directory ~/projects
``` 
mkdir ~/projects
```
3. cd into c:\projects
```
cd ~/projects
```
4. Clone the Github Blog repo to your local machine with the “git clone [Repo Name]” command. 

```
git clone https://github.com/Tsmith5151/Blog.git
```

or the original source code:

```
git clone https://github.com/PanosSakkos/personal-jekyll-theme.git
```

5.cd into the repo that you just cloned
```
cd Blog
```

6. Make sure that you have a GemFile with no file extension in the root of your repo with the following contents. Just to note, Github Pages supports very few jekyll plugins (i.e. jekyll-redirect-from).

```python
source 'https://rubygems.org'
gem 'github-pages'
gem 'jekyll-redirect-from'
```

7. From the Blog directory, run the following command to install gems listed in the Gemfile.

```
bundle install
```
8. Now we should have jekyll installed and we can test it out. In the terminal and in your blog repo directory, run the following command to tell jekyll to build and run the web site locally:

```
jekyll serve
````

Or you can build and serve your website by simply runningg:

````
./scripts/serve-production
````

#### Forked Repo

Source: [`personal-jekyll-theme`](https://github.com/PanosSakkos/personal-jekyll-theme)
