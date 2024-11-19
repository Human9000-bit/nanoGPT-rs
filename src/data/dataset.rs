use burn::data::dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset};
use derive_new::new;
use serde::Deserialize;

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    pub content: String,
}

pub struct DbPediaDataset {
    dataset: SqliteDataset<DbPediaItem>,
}

impl Dataset<TextGenerationItem> for DbPediaDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.content))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DbPediaDataset {
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }

    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct DvachItem {
    pub content: String,
}

pub struct DvachDataset {
    pub dataset: SqliteDataset<DvachItem>
}

impl DvachDataset {
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DvachItem> = HuggingfaceDatasetLoader::new("Vikhrmodels/2ch-24-09-2024-no-links")
            .dataset(split)
            .unwrap();
        
        Self { dataset }
    }
    
    pub fn train() -> Self {
        Self::new("train")
    }
}

impl Dataset<TextGenerationItem> for DvachDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.content))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
